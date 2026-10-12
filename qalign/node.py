import json
import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "QAlign requires OpenCV (cv2). Install opencv-python in the Python "
            "environment used by ComfyUI, then restart ComfyUI."
        )


def _tensor_to_rgb_u8(image):
    if image.ndim == 4:
        image = image[0]
    arr = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _rgb_u8_to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr[..., :3]).astype(np.float32) / 255.0).unsqueeze(0)


def _resize_to(rgb, width, height):
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _gray_feature(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _limit_translation(M, max_shift):
    M = M.astype(np.float32).copy()
    M[0, 2] = float(np.clip(M[0, 2], -max_shift, max_shift))
    M[1, 2] = float(np.clip(M[1, 2], -max_shift, max_shift))
    return M


def _estimate_global(reference, source, mode, max_shift, ecc_iterations, pyramid_levels,
                     max_scale_deviation, max_rotation_degrees, scale_mode="Full affine",
                     max_scale_deviation_x=0.12, max_scale_deviation_y=0.12):
    """
    Estimate a source-to-reference affine warp with coarse-to-fine ECC.
    Translation is carried between pyramid levels in full-resolution pixel units.
    OpenCV ECC warp is applied using WARP_INVERSE_MAP at warp time.
    """
    h, w = reference.shape[:2]
    ref_gray_full = _gray_feature(reference)
    src_gray_full = _gray_feature(source)
    report = {
        "global_method": mode,
        "global_success": False,
        "pyramid_levels_requested": int(pyramid_levels),
        "pyramid_levels_used": [],
    }

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    if mode in ("Translation", "Hybrid", "Affine"):
        # Phase correlation supplies a useful translation seed; it can fail when
        # appearance changes substantially, so a bounded zero seed remains valid.
        try:
            shift, response = cv2.phaseCorrelate(
                src_gray_full.astype(np.float32), ref_gray_full.astype(np.float32)
            )
            dx, dy = float(shift[0]), float(shift[1])
            if np.isfinite(dx) and np.isfinite(dy) and abs(dx) <= max_shift and abs(dy) <= max_shift:
                M[0, 2], M[1, 2] = dx, dy
            report["phase_shift_x"] = round(dx, 3)
            report["phase_shift_y"] = round(dy, 3)
            report["phase_response"] = round(float(response), 5)
        except Exception as exc:
            report["phase_error"] = str(exc)[:160]

    if mode == "Translation":
        report["global_success"] = True
        report["transform"] = [[round(float(v), 6) for v in row] for row in M.tolist()]
        return _limit_translation(M, max_shift), report

    # Construct coarse-to-fine levels, avoiding dimensions too small for ECC.
    scales = [2 ** i for i in range(max(0, int(pyramid_levels) - 1), -1, -1)]
    used = []
    last_score = None
    current = M.copy()
    for divisor in scales:
        lw, lh = max(64, w // divisor), max(64, h // divisor)
        # Don't upscale small images beyond full resolution.
        lw, lh = min(w, lw), min(h, lh)
        if (lw, lh) in used:
            continue
        used.append((lw, lh))
        ref_l = cv2.resize(ref_gray_full, (lw, lh), interpolation=cv2.INTER_AREA if lw < w else cv2.INTER_LINEAR)
        src_l = cv2.resize(src_gray_full, (lw, lh), interpolation=cv2.INTER_AREA if lw < w else cv2.INTER_LINEAR)
        sx, sy = lw / float(w), lh / float(h)
        warp = current.copy()
        warp[0, 2] *= sx
        warp[1, 2] *= sy
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(ecc_iterations), 1e-6)
        try:
            score, warp_l = cv2.findTransformECC(
                ref_l, src_l, warp, cv2.MOTION_AFFINE, criteria, None, 5
            )
            # Convert translation back to full-resolution coordinates.
            candidate = warp_l.astype(np.float32).copy()
            candidate[0, 2] /= sx
            candidate[1, 2] /= sy
            candidate = _limit_translation(candidate, max_shift)

            # Reject transforms that would be too extreme for a global registration.
            linear = candidate[:, :2]
            det = float(np.linalg.det(linear))
            scale_x = float(np.linalg.norm(linear[:, 0]))
            scale_y = float(np.linalg.norm(linear[:, 1]))
            angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
            limit_x = float(max_scale_deviation_x) if scale_mode == "Anisotropic X/Y" else float(max_scale_deviation)
            limit_y = float(max_scale_deviation_y) if scale_mode == "Anisotropic X/Y" else float(max_scale_deviation)
            valid = (
                det > 0.0
                and abs(scale_x - 1.0) <= limit_x
                and abs(scale_y - 1.0) <= limit_y
                and abs(angle) <= float(max_rotation_degrees)
            )
            if valid:
                # ECC estimates a full affine matrix. In Anisotropic X/Y mode,
                # retain independent axis scales and rotation but remove shear.
                # This makes the correction predictable for resize-induced drift.
                if scale_mode == "Anisotropic X/Y":
                    theta = np.radians(angle)
                    c, sn = float(np.cos(theta)), float(np.sin(theta))
                    candidate[0, 0] = c * scale_x
                    candidate[1, 0] = sn * scale_x
                    candidate[0, 1] = -sn * scale_y
                    candidate[1, 1] = c * scale_y
                current = candidate
                last_score = float(score)
                report["global_success"] = True
                report["ecc_score"] = round(float(score), 5)
                report["accepted_levels"] = report.get("accepted_levels", 0) + 1
                report["scale_x_estimate"] = round(scale_x, 6)
                report["scale_y_estimate"] = round(scale_y, 6)
                report["scale_mode"] = scale_mode
            else:
                report["rejected_transform_at"] = [lw, lh]
        except Exception as exc:
            report.setdefault("ecc_errors", []).append(f"{lw}x{lh}: {str(exc)[:120]}")

    report["pyramid_levels_used"] = [[int(a), int(b)] for a, b in used]
    report["global_score_final"] = round(last_score, 5) if last_score is not None else None
    report["transform"] = [[round(float(v), 6) for v in row] for row in current.tolist()]
    return current, report



def _estimate_automatic_global(reference, source, max_shift, max_scale_deviation,
                               max_rotation_degrees, scale_mode, max_scale_deviation_x,
                               max_scale_deviation_y, ecc_iterations, pyramid_levels):
    """Robust feature-based affine initialization, with ECC fallback.

    ORB matches are filtered with a ratio test and RANSAC. The resulting matrix
    maps reference coordinates to source coordinates, matching ECC's warp
    convention (used with WARP_INVERSE_MAP during rendering).
    """
    h, w = reference.shape[:2]
    ref_gray = _gray_feature(reference)
    src_gray = _gray_feature(source)
    report = {"global_method": "Automatic", "global_success": False,
              "automatic_estimator": "ORB + ratio test + RANSAC"}

    feature_matrix = None
    try:
        # Increase feature count for high-resolution photographs with changed subjects.
        orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=19, patchSize=31, fastThreshold=10)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        kp_src, des_src = orb.detectAndCompute(src_gray, None)
        report["reference_keypoints"] = len(kp_ref)
        report["source_keypoints"] = len(kp_src)
        if des_ref is not None and des_src is not None and len(kp_ref) >= 6 and len(kp_src) >= 6:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            pairs = matcher.knnMatch(des_src, des_ref, k=2)
            good = [m for pair in pairs if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.75 * n.distance]
            report["ratio_test_matches"] = len(good)
            if len(good) >= 6:
                # Spatially balance matches across the reference image so a dense
                # textured region cannot dominate the transform estimate. This is
                # not semantic subject detection; it simply improves image-wide coverage.
                grid_cols, grid_rows, per_cell_cap = 8, 8, 80
                buckets = {}
                for match in sorted(good, key=lambda m: m.distance):
                    x, y = kp_ref[match.trainIdx].pt
                    gx = min(grid_cols - 1, max(0, int(x / max(1, w) * grid_cols)))
                    gy = min(grid_rows - 1, max(0, int(y / max(1, h) * grid_rows)))
                    cell = (gx, gy)
                    bucket = buckets.setdefault(cell, [])
                    if len(bucket) < per_cell_cap:
                        bucket.append(match)
                balanced = [m for cell_matches in buckets.values() for m in cell_matches]
                report["spatial_match_grid"] = [grid_cols, grid_rows]
                report["spatially_balanced_matches"] = len(balanced)
                report["occupied_match_cells"] = len(buckets)
                src_pts = np.float32([kp_src[m.queryIdx].pt for m in balanced])
                ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in balanced])
                # ECC's warp is reference -> source.
                feature_matrix, inlier_mask = cv2.estimateAffine2D(
                    ref_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                    maxIters=5000, confidence=0.995, refineIters=20)
                inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
                report["ransac_inliers"] = inliers
                report["ransac_inlier_ratio"] = round(inliers / max(1, len(balanced)), 4)
    except Exception as exc:
        report["feature_estimation_error"] = str(exc)[:180]

    def acceptable(matrix):
        if matrix is None or not np.isfinite(matrix).all():
            return False
        linear = matrix[:, :2]
        det = float(np.linalg.det(linear))
        sx = float(np.linalg.norm(linear[:, 0]))
        sy = float(np.linalg.norm(linear[:, 1]))
        angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        limx = float(max_scale_deviation_x) if scale_mode == "Anisotropic X/Y" else float(max_scale_deviation)
        limy = float(max_scale_deviation_y) if scale_mode == "Anisotropic X/Y" else float(max_scale_deviation)
        return (det > 0 and abs(sx - 1) <= limx and abs(sy - 1) <= limy
                and abs(angle) <= float(max_rotation_degrees)
                and abs(float(matrix[0, 2])) <= max_shift
                and abs(float(matrix[1, 2])) <= max_shift)

    if acceptable(feature_matrix) and report.get("ransac_inliers", 0) >= 8:
        M = feature_matrix.astype(np.float32)
        linear = M[:, :2]
        sx = float(np.linalg.norm(linear[:, 0]))
        sy = float(np.linalg.norm(linear[:, 1]))
        angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        if scale_mode == "Anisotropic X/Y":
            theta = np.radians(angle)
            c, sn = float(np.cos(theta)), float(np.sin(theta))
            M[0, 0], M[1, 0] = c * sx, sn * sx
            M[0, 1], M[1, 1] = -sn * sy, c * sy
        report["global_success"] = True
        report["automatic_estimate_source"] = "feature RANSAC"
        report["scale_x_estimate"] = round(sx, 6)
        report["scale_y_estimate"] = round(sy, 6)
        report["rotation_degrees_estimate"] = round(angle, 4)
        report["translation_x_estimate_px"] = round(float(M[0, 2]), 3)
        report["translation_y_estimate_px"] = round(float(M[1, 2]), 3)
        report["transform"] = [[round(float(v), 6) for v in row] for row in M.tolist()]
        return M, report

    # When feature matches are weak or implausible, fall back to the existing
    # multi-scale ECC estimator rather than applying an untrusted transform.
    fallback, fallback_report = _estimate_global(
        reference, source, "Affine", max_shift, ecc_iterations, pyramid_levels,
        max_scale_deviation, max_rotation_degrees, scale_mode,
        max_scale_deviation_x, max_scale_deviation_y)
    report["automatic_estimate_source"] = "ECC fallback"
    report["automatic_fallback_reason"] = "Insufficient reliable feature matches or transform outside limits"
    report.update({f"fallback_{k}": v for k, v in fallback_report.items()})
    report["global_success"] = bool(fallback_report.get("global_success", False))
    report["transform"] = [[round(float(v), 6) for v in row] for row in fallback.tolist()]
    return fallback, report

def _make_edge_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 100)
    return cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1) > 0


def _local_warp(reference, aligned_source, strength, smoothness, max_flow, protect_background,
                subject_mask=None):
    """
    Estimate dense flow from reference to aligned source and warp the source.
    Flow is smoothed, bounded, strength-scaled, then applied. Reports both raw
    and applied flow magnitudes so diagnostics reflect the actual displacement.
    """
    h, w = reference.shape[:2]
    ref_gray = _gray_feature(reference)
    src_gray = _gray_feature(aligned_source)

    # Farneback is a coarse-to-fine dense optical-flow algorithm. More pyramid
    # levels and a larger window make it more tolerant of larger displacements.
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=5, winsize=31, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0
    )
    raw_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    k = max(3, int(smoothness) | 1)
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (k, k), 0)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (k, k), 0)
    smooth_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Bound the smoothed flow before strength is applied.
    limit = float(max_flow)
    scale = np.ones_like(smooth_mag, dtype=np.float32)
    over = smooth_mag > limit
    if limit <= 0:
        flow[:] = 0
    else:
        scale[over] = limit / np.maximum(smooth_mag[over], 1e-6)
        flow *= scale[..., None]
    flow *= float(strength) / 100.0

    mask_used = False
    if subject_mask is not None:
        mask = np.asarray(subject_mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(np.clip(mask, 0.0, 1.0), (0, 0), 3.0)
        flow *= mask[..., None]
        mask_used = True
    elif protect_background:
        # Conservative image-detail proxy, not semantic segmentation.
        ref_f = reference.astype(np.float32)
        blurred = cv2.GaussianBlur(ref_f, (0, 0), 11.0)
        detail = np.mean(np.abs(ref_f - blurred), axis=2)
        edges = _make_edge_mask(reference)
        proxy = (detail > 3.0) | edges
        proxy = cv2.dilate(proxy.astype(np.uint8), np.ones((17, 17), np.uint8), 1) > 0
        mask = cv2.GaussianBlur(proxy.astype(np.float32), (0, 0), 5.0)
        flow *= np.clip(mask, 0.0, 1.0)[..., None]

    applied_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        aligned_source, xx + flow[..., 0], yy + flow[..., 1],
        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
    )
    stats = {
        "local_flow_raw_mean_px": round(float(np.mean(raw_mag)), 4),
        "local_flow_raw_p95_px": round(float(np.percentile(raw_mag, 95)), 4),
        "local_flow_raw_max_px": round(float(np.max(raw_mag)), 4),
        "local_flow_smoothed_max_px": round(float(np.max(smooth_mag)), 4),
        "local_flow_applied_mean_px": round(float(np.mean(applied_mag)), 4),
        "local_flow_applied_p95_px": round(float(np.percentile(applied_mag, 95)), 4),
        "local_flow_applied_max_px": round(float(np.max(applied_mag)), 4),
        "local_strength_percent": int(strength),
        "local_flow_limit_px": float(max_flow),
        "background_protection": bool(protect_background),
        "subject_mask_used_for_flow": mask_used,
    }
    return warped, stats


class QAlignImageAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "source": ("IMAGE",),
                "alignment_mode": (["Automatic", "Hybrid", "Translation", "Affine", "Local only"],),
                "local_correction": ("BOOLEAN", {"default": True}),
                "correction_strength": ("INT", {"default": 100, "min": 0, "max": 100, "step": 5}),
                "max_global_shift": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "max_local_flow": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "flow_smoothing": ("INT", {"default": 81, "min": 3, "max": 301, "step": 2}),
                "protect_background": ("BOOLEAN", {"default": False}),
                "preserve_source_size": ("BOOLEAN", {"default": False}),
                "diagnostic_overlay": ("BOOLEAN", {"default": True}),
                "pyramid_levels": ("INT", {"default": 4, "min": 1, "max": 6, "step": 1}),
                "ecc_iterations": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "max_scale_deviation": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_rotation_degrees": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 90.0, "step": 1.0}),
                "scale_mode": (["Full affine", "Anisotropic X/Y"], {"default": "Anisotropic X/Y"}),
                "max_scale_deviation_x": ("FLOAT", {"default": 0.08, "min": 0.005, "max": 1.0, "step": 0.005}),
                "max_scale_deviation_y": ("FLOAT", {"default": 0.08, "min": 0.005, "max": 1.0, "step": 0.005}),
            },
            "optional": {
                "subject_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_image", "overlay_preview", "difference_preview", "alignment_report")
    FUNCTION = "align"
    CATEGORY = "QAlign"

    def align(self, reference, source, alignment_mode="Affine", local_correction=True,
              correction_strength=100, max_global_shift=256, max_local_flow=16.0,
              flow_smoothing=81, protect_background=False, preserve_source_size=False,
              diagnostic_overlay=True, pyramid_levels=4, ecc_iterations=100,
              max_scale_deviation=0.12, max_rotation_degrees=12.0, subject_mask=None,
              scale_mode="Anisotropic X/Y", max_scale_deviation_x=0.08,
              max_scale_deviation_y=0.08):
        _require_cv2()
        ref = _tensor_to_rgb_u8(reference)
        src_original = _tensor_to_rgb_u8(source)
        src = src_original.copy()
        original_src_size = (src.shape[1], src.shape[0])

        if preserve_source_size:
            ref_work = _resize_to(ref, src.shape[1], src.shape[0])
        else:
            src = _resize_to(src, ref.shape[1], ref.shape[0])
            ref_work = ref.copy()
        h, w = src.shape[:2]
        ref_work = _resize_to(ref_work, w, h)

        report = {
            "node": "QAlign",
            "version": "0.5.0",
            "reference_size": [int(ref.shape[1]), int(ref.shape[0])],
            "source_size": [int(original_src_size[0]), int(original_src_size[1])],
            "working_size": [int(w), int(h)],
            "alignment_mode": alignment_mode,
            "note": "Alignment estimates geometry; lighting and texture differences are not alignment errors.",
            "global_shift_limit_px": int(max_global_shift),
            "local_flow_limit_px": float(max_local_flow),
        }

        mask_np = None
        if subject_mask is not None:
            mask = subject_mask
            if mask.ndim == 3:
                mask = mask[0]
            mask_np = mask.detach().to("cpu", dtype=torch.float32).numpy()
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_np = np.clip(mask_np, 0.0, 1.0)

        if alignment_mode == "Local only":
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            global_report = {"global_method": "none", "global_success": True, "transform": M.tolist()}
        elif alignment_mode == "Automatic":
            M, global_report = _estimate_automatic_global(
                ref_work, src, int(max_global_shift), float(max_scale_deviation),
                float(max_rotation_degrees), str(scale_mode),
                float(max_scale_deviation_x), float(max_scale_deviation_y),
                int(ecc_iterations), int(pyramid_levels)
            )
        else:
            method = "Translation" if alignment_mode == "Translation" else "Affine"
            M, global_report = _estimate_global(
                ref_work, src, method, int(max_global_shift), int(ecc_iterations),
                int(pyramid_levels), float(max_scale_deviation), float(max_rotation_degrees),
                str(scale_mode), float(max_scale_deviation_x), float(max_scale_deviation_y)
            )
        report.update(global_report)
        report["scale_mode_requested"] = str(scale_mode)
        report["max_scale_deviation_x"] = float(max_scale_deviation_x)
        report["max_scale_deviation_y"] = float(max_scale_deviation_y)

        # ECC's matrix maps reference/template positions to source positions;
        # WARP_INVERSE_MAP is therefore required for source-to-reference alignment.
        aligned = cv2.warpAffine(
            src, M, (w, h),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101
        )

        local_report = {"local_correction_applied": False}
        if local_correction and correction_strength > 0 and max_local_flow > 0:
            aligned, local_report = _local_warp(
                ref_work, aligned, correction_strength, flow_smoothing, max_local_flow,
                bool(protect_background), mask_np
            )
            local_report["local_correction_applied"] = True
        report.update(local_report)

        # Optional mask also controls where corrected output is retained. Outside the
        # mask, keep the globally aligned image rather than the original unaligned source.
        if mask_np is not None:
            soft = cv2.GaussianBlur(mask_np.astype(np.float32), (0, 0), 1.5)[..., None]
            globally_aligned = cv2.warpAffine(
                src, M, (w, h),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101
            )
            aligned = np.clip(
                aligned.astype(np.float32) * soft +
                globally_aligned.astype(np.float32) * (1.0 - soft),
                0, 255
            ).astype(np.uint8)
            report["subject_mask_used"] = True
        else:
            report["subject_mask_used"] = False

        diff = cv2.absdiff(ref_work, aligned)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        if np.max(diff_gray) > 0:
            diff_gray = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        diff_preview = cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(ref_work, 0.5, aligned, 0.5, 0.0) if diagnostic_overlay else aligned.copy()

        report["mean_absolute_difference"] = round(float(np.mean(np.abs(
            ref_work.astype(np.float32) - aligned.astype(np.float32)
        ))), 4)
        report["interpretation"] = (
            "Difference includes intentional appearance changes. Use overlay and structural "
            "landmarks to judge alignment; it is not a pure alignment score."
        )
        return (
            _rgb_u8_to_tensor(aligned),
            _rgb_u8_to_tensor(overlay),
            _rgb_u8_to_tensor(diff_preview),
            json.dumps(report, indent=2),
        )


NODE_CLASS_MAPPINGS = {"QAlignImageAlign": QAlignImageAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"QAlignImageAlign": "QAlign"}

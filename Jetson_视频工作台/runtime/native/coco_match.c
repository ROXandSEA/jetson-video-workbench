/* COCO greedy bbox matching kernel; no fast-math, no metric approximations.
 * Build: gcc -O3 -fPIC -shared runtime/native/coco_match.c -o runtime/native/coco_match.so
 * The ordering/crowd rules follow pycocotools (BSD-2-Clause), see native/README.md.
 */
#include <stddef.h>
#include <stdint.h>

int coco_match_abi(void) { return 2; }

void coco_match(size_t thresholds, size_t detections, size_t targets,
                const double *ious, const double *levels,
                const uint8_t *crowd, const uint8_t *ignored,
                const double *detection_ids, const double *target_ids,
                double *dt_matches, double *gt_matches, uint8_t *dt_ignore) {
    for (size_t t = 0; t < thresholds; ++t) {
        for (size_t d = 0; d < detections; ++d) {
            double best = levels[t] < 1.0-1e-10 ? levels[t] : 1.0-1e-10;
            ptrdiff_t match = -1;
            for (size_t g = 0; g < targets; ++g) {
                if (gt_matches[t*targets+g] > 0 && !crowd[g]) continue;
                if (match >= 0 && !ignored[match] && ignored[g]) break;
                if (ious[d*targets+g] < best) continue;
                best = ious[d*targets+g];
                match = (ptrdiff_t)g;
            }
            if (match < 0) continue;
            dt_ignore[t*detections+d] = ignored[match];
            dt_matches[t*detections+d] = (double)target_ids[match];
            gt_matches[t*targets+(size_t)match] = (double)detection_ids[d];
        }
    }
}

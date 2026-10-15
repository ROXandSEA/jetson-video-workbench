/* Contiguous YOLO channel reduction and RGB letterbox packing. No fast-math. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int frame_ops_abi(void) { return 3; }

int yolo_candidates(const float *p, int n, int nc, float threshold,
                    float *scores, int32_t *classes, uint8_t *valid,
                    int32_t *indices) {
    for (int i = 0; i < n; ++i) {
        scores[i] = p[4*n+i]; classes[i] = 0;
        valid[i] = isfinite(p[i]) && isfinite(p[n+i]) &&
                   isfinite(p[2*n+i]) && isfinite(p[3*n+i]) &&
                   p[2*n+i] > 0 && p[3*n+i] > 0;
    }
    for (int c = 0; c < nc; ++c) {
        const float *row = p+(4+c)*n;
        for (int i = 0; i < n; ++i) {
            if (!isfinite(row[i])) valid[i] = 0;
            if (row[i] > scores[i]) { scores[i] = row[i]; classes[i] = c; }
        }
    }
    int count = 0;
    for (int i = 0; i < n; ++i)
        if (valid[i] && scores[i] >= threshold) indices[count++] = i;
    return count;
}

void pack_rgb(const uint8_t *bgr, int h, int w, int out_h, int out_w, int top, int left,
              float *output) {
    const int plane = out_h*out_w;
    float values[256];
    for (int i = 0; i < 256; ++i) values[i] = (float)i/255.0f;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < out_h; ++y) {
            float *dest = output+c*plane+y*out_w;
            if (y < top || y >= top+h) {
                for (int x = 0; x < out_w; ++x) dest[x] = values[114];
                continue;
            }
            for (int x = 0; x < left; ++x) dest[x] = values[114];
            const uint8_t *src = bgr+(y-top)*w*3+(2-c);
            for (int x = 0; x < w; ++x) dest[left+x] = values[src[x*3]];
            for (int x = left+w; x < out_w; ++x) dest[x] = values[114];
        }
}

/* Float32 -> double -> 17 digits preserves Python .tolist() JSON values.
 * ctypes releases the GIL while this formatting loop executes. */
int detection_json(const float *rows,int count,unsigned long long frame,char *out,int capacity) {
    int used=snprintf(out,capacity,"{\"frame\":%llu,\"detections\":[",frame);
    if(used<0||used>=capacity)return -1;
    for(int i=0;i<count;++i) {
        int n=snprintf(out+used,capacity-used,"%s[",i?",":"");
        if(n<0||n>=capacity-used)return -1;
        used+=n;
        for(int c=0;c<6;++c) {
            double value=(double)rows[i*6+c];if(!isfinite(value))return -1;
            n=snprintf(out+used,capacity-used,"%s%.17g",c?",":"",value);
            if(n<0||n>=capacity-used)return -1;
            used+=n;
        }
        if(used+1>=capacity)return -1;
        out[used++]=']';
    }
    if(used+4>capacity)return -1;
    out[used++]=']';out[used++]='}';out[used++]='\n';out[used]=0;
    return used;
}

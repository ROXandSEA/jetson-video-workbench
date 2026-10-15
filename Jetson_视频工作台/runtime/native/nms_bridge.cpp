/* Greedy classwise NMS matching OpenCV 4.5.4's floating-point overlap rule.
 * Stable score ties and original class-group ordering are retained. */
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>
struct Box { double x,y,w,h; };
extern "C" const char *nms_reference_version() { return "4.5.4"; }
extern "C" int nms_abi() { return 2; }
static float overlap(const Box &a,const Box &b) {
    double total=a.w*a.h+b.w*b.h;
    if(total<=std::numeric_limits<double>::epsilon())return 1.0f;
    double w=std::max(0.0,std::min(a.x+a.w,b.x+b.w)-std::max(a.x,b.x));
    double h=std::max(0.0,std::min(a.y+a.h,b.y+b.h)-std::max(a.y,b.y));
    double inter=w*h;
    // Preserve the double -> float distance -> float overlap rounding.
    return 1.0f-(float)(1.0-inter/(total-inter));
}
extern "C" int classwise_nms(const float *boxes,const float *scores,const int32_t *classes,
                             int count,int nc,float threshold,int limit,int32_t *result) {
    try {
        int total=0;
        for(int c=0;c<nc;++c) {
            std::vector<Box> rectangles;
            std::vector<int> positions,kept;
            for(int i=0;i<count;++i)if(classes[i]==c) {
                const float *b=boxes+4*i;
                float w=b[2]-b[0],h=b[3]-b[1];
                rectangles.push_back({(double)b[0],(double)b[1],(double)w,(double)h});
                positions.push_back(i);
            }
            if(rectangles.empty())continue;
            std::vector<int> order;
            for(int i=0;i<(int)positions.size();++i)if(scores[positions[i]]>0)order.push_back(i);
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[positions[a]]>scores[positions[b]];});
            for(int index:order) {
                bool accept=true;
                for(int earlier:kept)if(!(overlap(rectangles[index],rectangles[earlier])<=threshold)){accept=false;break;}
                if(accept)kept.push_back(index);
                // Later entries in this class cannot enter the global top limit.
                if((int)kept.size()>=limit)break;
            }
            for(int index:kept)result[total++]=positions[index];
        }
        return total;
    } catch(...) { return -1; }
}

//this is aa test
//若a[i]>=0,先对a排序，利用优先队列求解
#include <bits/stdc++.h>
using namespace std;
long long find_k_min(vector<int> &nums,int &n,int &k){
    using PLL = pair<long long, long long>;
    priority_queue<PLL, vector<PLL>, greater<PLL> > q;
    long long sum = 0,mx = 0;
    for( int i = 0; i < nums.size(); ++i){
        if(nums[i] < 0) sum += nums[i];
        nums[i] = abs(nums[i]);
        mx += nums[i];
    }

    if(k == 1){
        return mx + sum;
    }
    k-=2;
    sort(nums.begin(), nums.end()); 
    q.push({nums[0], 0});
    while(k--){
        auto u= q.top();q.pop();
        //cout<< u.first <<endl;
        if( u.second + 1 >= nums.size() ) continue;
        q.push({u.first +nums[u.second + 1] -nums[u.second] , u.second+1});
        q.push({u.first + nums[u.second + 1], u.second +1 });
    }
    return (-q.top().first + mx + sum);
};
int main(){
	
}
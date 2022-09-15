//this is aa test
#include <bits/stdc++.h>

using namespace std;
int solve(vector<int> &nums, int &k){
	int l = 0, r = 0;
	for(int i = 0; i < nums.size(); ++i){
		r += nums[i];
	}
	int res = -1;
	auto check = [&](int mid){
		int l = 0, sum = 0 , res = 0;
		for(int i = 0; i < nums.size(); ++i){
			if(sum + nums[i] > k){
				res += (1 + i - l) * (i - l) / 2;
			}
			while(sum > k){
				sum -= nums[l];
				l ++;
			}
		}
		return res;
	};
	while(l <= r){
		int mid = (l + r) / 2;
		if(check(mid) >= k){
			r = mid - 1;
			res = mid;
		}else{
			l = mid + 1;
		}
	}
	return res;
}
int main(){

	return 0;
}
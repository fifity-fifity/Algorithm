#####背包+树形dp

多重背包
for(int i=1;i<=n;++i)//物品件数
{
    int num=cnt[i]; //第i件物品的数量
    for(int k=1;num>0;k*=2) //二进制优化，将n件物品分成logn件
    {
        if(k>num)
            k=num;
        num-=k;
        for(int j=V;j>=w[i]*k;--j)
        {
            dp[j]=max(dp[j],dp[j-w[i]*k]+val[i]*k);
        }
    }
}


分组背包：每组物品只拿一个
for(int k=1;k<=cnt;++k)//枚举组数
{
    for(int j=V;j>=0;j--)//枚举容量
    {
        for(int i=1;i<=n;++i)//枚举物品
        {
            if(j>w[i]) //保证合理性
                dp[j]=max(dp[j],dp[j-w]+val);
        }
    }
}

二维费用
for(int k=1;k<=n;++k)
{
   for(int i=m;i>=mi;i--) //枚举价值
   {
       for(int j=t;j>=tj;j--) //枚举时间
        dp[i][j]=max(dp[i][j],dp[i-mi][j-tj]+val);
   }
}

二维费用：https://atcoder.jp/contests/abc219/tasks/abc219_d

有依赖的背包(树形dp+分组背包)
O(n*n)// (e.g.  https://codeforces.com/contest/816/problem/E)
void dfs(int u)
{
    dp[u][0]=0;
    dp[u][1]=a[u].val;//更新当前花费一个节点的价值
    for(auto v:vec[u])//遍历u的子节点
    {
        dfs(v);//更新子节点
        for(int j=sz[u];j>=1;j--)//j>=1表示根节点物品一定要选
        {
            for(int k=0;k<=sz[v];++k)
            {
                dp[u][j+k]=max(dp[u][j+k],dp[u][j]+dp[v][k]);
            }
        }
        sz[u]+=sz[v];
    }
}
利用dfs序优化O(n*v) （金明的预算）
void dfs(int u)
{
    cnt[u]=1;//cnt记录节点u的大小
    for(auto x:vec[u])
    {
        dfs(x);
        cnt[u]+=cnt[x];
    }
    num[++tot]=u;//num记录一个dfs序列
}
for(int i=1;i<=n;++i)
{
       for(int j=0;j<=v;j++)
       {
             if(j>=a[num[i]].w) 
                dp[i][j]=max(dp[i-cnt[num[i]]][j],dp[i-1][j-a[num[i]].w]+a[num[i]].val);//选该节点
             else 
                dp[i][j]=dp[i-cnt[num[i]]][j];//不选该节点，由子树转移
        }
 }


通配符匹配：
f[i][j]=f[i][j-1] || f[i-1][j]; if(t[j]=='*');
//f[i][j-1]表示当前t[j]=='*'匹配空字符，
//f[i-1][j]表示t[j]=='*'再多匹配一个字符s[i]
f[i][j]=f[i-1]j-1];if(s[i]==t[j] || t[j]=='?')

最长上升子序列

单调队列优化dp：连续的m个或者每隔m个必须选择一个
    int q[N+5];//单调递增栈求最小值

    int hh=0,tt=0;
    for(int i=1;i<=n;++i)
    {
        while(hh<=tt&&q[hh]<i-k-1) hh++;
        dp[i]=dp[q[hh]]+a[i];
        while(hh<=tt&&dp[q[tt]]>dp[i]) tt--;
        q[++tt]=i;
    }
    int ans=inff;
    for(int i=n-k;i<=n;++i)
        ans=min(ans,dp[i]);
    cout<<sum-ans<<endl;
//最多选择连续的k个并使得结果最大

用f[i][j]表示前i块水晶搭建的双塔高度差值为j时较高的塔的高度

这样f[i][j]的值就会存在4种情况：
1.这块水晶不放                           f[i-1][j]                
2.放在矮的塔上且没超过高塔     f[i-1][j+a[i]] //原来差值为j+a[i]，把a[i]放矮的上面        
3.放在高的塔上                           f[i-1][j-a[i]]+a[i]   //原来差值为j-a[i],把a[i]放高的上
4.放在矮的塔上但超过了高塔     f[i-1][a[i]-j]+j       //原来差值为a[i]-j，把塔放矮的上
第4种情况
      设高塔高x+h,矮塔高x;
      x+a[i]-(x+h)=a[i]-h=j
     所以h=a[i]-j，新塔高度为x+a[i]，而原来高塔高x+a[i]-j，故在高塔上加j即可
     综上: dp[i][j]=x+a[i];代入x即可


#####并查集
```cpp
初始化
void init()
{
     for(int i=0;i<=N;++i)  fa[i]=i;
}
查询，可以简单证明压缩路径使得树深不超过2	(1)
int _find(int x)
{
    return x==fa[x]? x:fa[x]=_find(fa[x]);
}

void join(int u,int v)
{
    fa[j_find(u)]=_find(v);
}

（二）：种类并查集
开k倍数组，不同区间记录不同关系。

（三）带权并查集，基于食物链

更新权值+压缩路径
int _find(int x)
{
    if(x==fa[x]) return x;
    int temp=_find(fa[x]);		//压缩父节点路径并更新fa[x]到根节点权值	,不能省略，@(1)
    dis[x]=(dis[x]+dis[fa[x]])%3;	//更新当前点x的权值为x->fa[x]+fa[x]->根节点
    return fa[x]=temp//=_find(fa[x]);		//利用temp减少递归cen数
}
void join(int u,int v,int x)
{
    int fu=_find(u),fv=_find(v);
    if(fu!=fv)//不在一个集合中
    {
        fa[fu]=fv;			//！！直接合并父节点
        dis[fu]=(dis[v]-dis[u]+x+2)%3;	//向量法更新fu的权值
    }
}
//区间合并例题
const int N=16;
int mod=1e9+7;
int fa[N+5],dis[N+5],ans;
int _find(int x)
{
    if(x==fa[x]) return x;
    int temp=_find(fa[x]);		//压缩父节点路径并更新fa[x]到根节点权值	,不能省略，@(1)
    dis[x]=dis[x]+dis[fa[x]];	//更新当前点x的权值为x->fa[x]+fa[x]->根节点
    return fa[x]=temp ;//=_find(fa[x]);		//利用temp减少递归cen数
}
void join(int u,int v,int x)
{
    int fu=_find(u),fv=_find(v);
    if(fu!=fv)//不在一个集合中
    {
        fa[fu]=fv;			//！！直接合并父节点
        dis[fu]=dis[v]-dis[u]+x;	//向量法更新fu的权值
    }
    else if(dis[u]-dis[v]!=x)
    {
        //cout<<v<<" "<<u<<" "<<dis[v]<<" "<<dis[u]<<endl;
        ans++;
    }
}
signed main()
{
    IOS;
    int n,m;cin>>n>>m;
    for(int i=0;i<=n;++i) fa[i]=i; //从0开始
    for(int i=1;i<=m;++i)
    {
        int l,r,val;cin>>l>>r>>val;
        join(l-1,r,val);//已知[1,3]和[6,10]时，显然已经知道[4,5]的值,l-1则是(0,3]和(5,10]使得[4,5]
        //与原来集合产生关联
    }
    cout<<ans<<endl;
    return 0;
}
```
#####超级源点+反向建图
```cpp
给定一个数组a[n]，其中i可以到i-a[i]和i+a[i]，问i到与a[i]不同奇偶性的点的最短路径长度
#include <bits/stdc++.h>
#define PII pair<int,int>
#define inf 0x3f3f3f3f
//#define int long long
using namespace std;
const int N=2e5;
int a[N+5],n,ans[N+5],dis[N+5],vis[N+5];
vector<int> g[N+5];
void dij(int s){
    priority_queue<PII,vector<PII>,greater<PII> > q;
    q.push({0,s});
    memset(dis,0x3f,sizeof dis);
    memset(vis,0,sizeof vis);
    dis[s]=0;
    while(q.size())
    {
        int u=q.top().second;
        int len=q.top().first;
        q.pop();
        if(vis[u]) continue;
        vis[u]=1;
        for(auto x:g[u]){
            if(dis[x]>1+len){
                dis[x]=1+len;
                q.push({dis[x],x});
            }
        }
    }
}
signed main()
{
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);std::cout.tie(0);

    cin>>n;
    for(int i=1;i<=n;++i)
        cin>>a[i];
    for(int i=1;i<=n;++i){
        if(i-a[i]>=1) g[i-a[i]].push_back(i);
        if(i+a[i]<=n) g[i+a[i]].push_back(i);
        if(a[i]&1) g[n+1].push_back(i);
        else g[n+2].push_back(i);
    }
    dij(n+1);
    for(int i=1;i<=n;++i){
        if(a[i]%2==0) ans[i]=dis[i]-1;
    }
    dij(n+2);
    for(int i=1;i<=n;++i){
        if(a[i]&1) ans[i]=dis[i]-1;
    }
    for(int i=1;i<=n;++i)
        if(ans[i]<1e9)
            cout<<ans[i]<<" ";
        else
            cout<<-1<<" ";
    cout<<endl;
    return 0;
}

/*
20 20 4 0
*/
```
#####求解严格第k大的背包
```cpp
#include<bits/stdc++.h>
#define int long long
#define endl '\n'
using namespace std;

const int N=1e5;
int dp[1005][55],a[55],b[55];
int val[N+5],w[N+5];
signed main()
{
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    int t;cin>>t;
    while(t--)
    {
        int n,v,k;cin>>n>>v>>k;
        memset(dp,0,sizeof dp);
        for(int i=1;i<=n;++i)
            cin>>val[i];
        for(int i=1;i<=n;++i)
            cin>>w[i];
        int ans=0;
        for(int i=1;i<=n;++i)
        {
            for(int j=v;j>=w[i];j--)
            {
                int p=1;
                for(p=1;p<=k;++p)
                {
                    a[p]=dp[j][p];
                    b[p]=dp[j-w[i]][p]+val[i];
                }
                a[p]=b[p]=-1;
                int c=1,x=1,y=1;
                while(c<=k&&(a[x]!=-1||b[y]!=-1))
                {
                    if(a[x]>b[y])
                        dp[j][c]=a[x],x++;
                    else
                        dp[j][c]=b[y],y++;
                    if(dp[j][c]!=dp[j][c-1])
                        c++;
                }
                ans=max(ans,dp[j][k]);
            }
        }
        cout<<ans<<endl;
    }

    return 0;
}
```
#####二分图染色,最大二分图匹配
```cpp

vector <int> G[N+5];
int col[N+5];
void dfs(int s,int c)
{
    col[s]=c;
    for(auto x:G[s])
    {
        if(col[x]==c)
            fl=0;
        else if(col[x]==-1)
            dfs(x,c^1);
    }
}
mem(col,-1);
for(int i=1;i<=n;++i)
{
    if(!col[i]) dfs(i,1);
}

最大二分图匹配
int g[505][505],vis[505],match[505];

int dfs(int x)
{
    for(int i=1;i<=m;++i)
    {
        if(g[x][i]){
            if(vis[i]) continue;
            vis[i]=1;
            if(!match[i]||dfs(match[i]))
            {
                match[i]=x;
                return 1;
            }
        }
    }
    return 0;
}
for(int i=1;i<=n;++i)
{
    mem(vis,0);
    ans+=dfs(i);
}
```
#####克鲁斯卡尔重构树
```cpp

//克鲁斯卡尔重构树性质
①如果我们要在一张图上找到一条u,v的路径，使得这条路径上的最大值最小，那么，用Kruskal重构树很容易实现：
我们按照边权从小到大来排序，然后重构树，最后的答案就是Kruskal重构树上LCA(u,v)的点权。
②是一颗一cnt为根的有根树，最小生成树重构时大根堆
③
//克鲁斯卡尔重构树+lca倍增查询
    for(int i=1;i<=m;++i)
    {
        int u=e[i].u,v=e[i].v,len=e[i].len;
        int fu=Find(u),fv=Find(v);
        if(fu!=fv)
        {
            ++cnt;
            b[cnt]=len;//新的点的点权为原来的边权
            g[cnt].push_back(fu);
            g[cnt].push_back(fv);
            //g[fu].push_back(cnt);
            //g[fv].push_back(cnt);
            faa[fu]=faa[fv]=cnt;
        }
    }
    dfs(cnt,0);//重构树是课有根树,cnt为树的根
```

#####欧拉回路与哈密顿回路
```cpp
欧拉回路存在：无重边，度数全为偶数或只有2个奇点（欧拉路径）。
dfs：O(v+e)时间内求路径。
//欧拉回路，cnt==n+1表示存在欧拉回路，经过每条边1次回到起点
vector<int> G[N+5];
int in[N+5],out[N+5],vis[N+5];
int path[N+5],cnt;
map<PII,int> mp;
void dfs(int u)
{
    for(auto i:G[u])
    {
        if(!mp[{u,i}])
        {
           mp[{u,i}]=1;
           dfs(i);
           path[cnt++]=i;
        }
    }
}

//哈密顿回路，经过每个点一次回到起点
int dp[1<<N][N+5];//状态压缩
int G[N+5][N+5],n,m;//初始化dp和G为inf,需特殊考虑重边
int solve()
{
    int states=(1<<n);//所有状态
    mem(dp,inf);
    dp[1][0]=0;
    for(int state=0;state<states;++state)  //枚举状态
    {
        for(int bit=0;bit<n;++bit)  //枚举选择第几个点
        {
            if(((state>>bit)&1)==1)  //如果state状态bit点已经被选择，则考虑该状态如何转移来
            {
                int mask=(1<<bit);
                int prev_state=(state^(1<<bit));  //上一个状态
                for(int prev=0;prev<n;++prev)  //枚举上一个状态下从哪个点转移到state状态
                {
                    if(((prev_state>>prev)&1)==1) //如果该点可以选择
                    {
                        dp[state][bit]=min(dp[state][bit],dp[prev_state][prev]+G[prev][bit]);//状态转移
                    }
                }
            }
        }
    }
    int res=inf;
    for(int i=0;i<n;++i)
        res=min(res,dp[states-1][i]+G[i][0]);//遍历所有点且回到点i的极值
    if(res>=inf) return -1;
    return res;
}
```
#####区间dp
```cpp
合并石子
线型
for(int len=1;len<=n;++len)
    {
        for(int l=1;l+len-1<=n;++l)
        {
            int r=l+len-1;
            for(int k=l;k<r;++k)
            {
                dp[l][r]=min(dp[l][r],dp[l][k]+dp[k+1][r]+sum[r]-sum[l-1]);
            }
        }
    }

环型
    for(int len=1;len<=2*n;++len) //len<=n | | len<=2*n
    {
        for(int l=1;l+len-1<=2*n;++l)
        {
            int r=l+len-1;
            for(int k=l;k<r;++k)
            {
                dp[l][r]=min(dp[l][r],dp[l][k]+dp[k+1][r]+sum[r]-sum[l-1]);
            }
        }
    }
int ans=inf;
for(int i=1;i<=n;++i) ans=min(ans,dp[i][i+n-1];

矩阵最优连乘//dp[l][r]表示l+1,l+1..l+r-1区间块的解，且即a[l]，a[r]不被选
        mem(dp,0);
        for(int len=3;len<=n;++len)//长度初始为3
        {
            for(int l=1;l+len-1<=n;++l)
            {
                int r=l+len-1;
                dp[l][r]=inf;
                for(int k=l;k<=r;++k)
                {
                    dp[l][r]=min(dp[l][r],dp[l][k]+dp[k][r]+a[l]*a[k]*a[r]);//a[l],a[k],a[r]不被选
                    //那么合并的cost为a[l]*a[k]*a[r]
                }
            }
        }
        cout<<dp[1][n]<<endl;
//区间dp不一定是三重循环，有些可能是二重循环
```

#####树链剖分lca
```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=1e6;
int n,m,s,dep[N+5],fa[N+5],son[N+5],top[N+5],sz[N+5];
vector<int> vec[N+5];
//dfs1求所有节点的size, depth, fa,son(重链节点)
int dfs1(int now,int pre,int d)
{
    dep[now]=d;
    fa[now]=pre;
    sz[now]=1;  //*
    int mx=-1; 
    for(auto x:vec[now]){
        if(x==pre) continue;
        sz[now]+=dfs1(x,now,d+1);
        if(sz[x]>mx)
        {
            mx=sz[x];
            son[now]=x;
        }
    }
    return sz[now];
}

//dfs2求每个节点i沿着重链能往上跳的最高的节点top[i]
void dfs2(int x,int pre)
{
    if(x==son[fa[x]]) top[x]=top[fa[x]];//当前点是重子节点
    else top[x]=x;  //当前点是轻子节点
    for(auto xx:vec[x]){
        if(xx==pre) continue;
        dfs2(xx,x);
    }
}
int lca(int u,int v)
{

    while(top[u]!=top[v])
    {
        if(dep[top[u]]>dep[top[v]])
            u=fa[top[u]];//*
        else
            v=fa[top[v]];

    }
    return dep[u]<dep[v]? u:v;
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);
    cin>>n>>m>>s;
    for(int i=1;i<=n-1;++i)
    {
        int u,v;cin>>u>>v;
        vec[u].push_back(v);
        vec[v].push_back(u);
    }
    dfs1(s,0,1);
    dfs2(s,0);
    for(int i=1;i<=m;++i)
    {
        int u,v;cin>>u>>v;
        cout<<lca(u,v)<<endl;
    }
    return 0;
}
```
#####树上启发式合并
```cpp
//每个点有一种颜色
//求每个节点的子树的占领点的点权和
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N=1e5;
const int mod=1e9+7;
int n,sz[N+5],son[N+5];;
int dfn=0,sum=0,mx=0,cnt[N+5],ans[N+5],col[N+5];
int skip;
vector<int> g[N+5];

int dfs(int now,int pre){
    sz[now]=1;
    int mx=-1;
    for(auto x:g[now]){
        if(x==pre) continue;
        dfs(x,now);
        sz[now]+=sz[x];
        if(sz[x]>mx){
            mx=sz[x];
            son[now]=x;
        }
    }
    return sz[now];
}
void add(int now,int pre){
    cnt[col[now]]++;
    if(cnt[col[now]]>mx){
       mx=cnt[col[now]];
       sum=col[now];
    }
    else if(cnt[col[now]]==mx){
        sum+=col[now];
    }
    for(auto &x:g[now]){
        if(x==pre||x==skip)
            continue;
        add(x,now);
    }
}
void del(int now,int pre){
    cnt[col[now]]--;
    for(auto &x:g[now]){
        if(x==pre||x==skip)
            continue;
        del(x,now);
    }
}
void solve(int now,int pre,int keep){
    for(auto &x:g[now]){
        if(x==pre||x==son[now])
            continue;
        solve(x,now,0);
    }
    if(son[now]){
        solve(son[now],now,1);
        skip=son[now];
    }
    add(now,pre);
    skip=0;
    ans[now]=sum;
    if(keep==0){
        del(now,pre);
        sum=mx=0;
    }
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);
    cin>>n;
    for(int i=1;i<=n;++i)
        cin>>col[i];
    for(int i=1;i<=n-1;++i){
        int u,v;cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs(1,0);
    solve(1,0,1);
    for(int i=1;i<=n;++i)
        cout<<ans[i]<<" "
    cout<<endl;
    return 0;
}
```
#####树上线段树
```cpp
//单点修改维护+树上两点间的权值和以及最大值
//优先更新重子节点

#include <bits/stdc++.h>
#define PII pair<int,int>
#define int long long
#define LL long long
using namespace std;
const int N=1e5;
int fa[N+5],son[N+5],sz[N+5],dep[N+5];
int id[N+5],top[N+5],ff[N+5],cnt; //id记录一个dfs序
int a[N+5];
vector<int> vec[N+5];
struct node{
    int l,r,mx,sum;
}tree[4*N+50];
void push_up(int i)
{
    tree[i].mx=max(tree[2*i].mx,tree[2*i+1].mx);
    tree[i].sum=tree[i*2].sum+tree[i*2+1].sum;
}
void build(int i,int l,int r)
{
    tree[i].l=l;
    tree[i].r=r;
    tree[i].mx=-1e15;
    tree[i].sum=0;
    if(l==r)
    {
        return ;
    }
    int mid=(l+r)/2;
    build(i*2,l,mid);
    build(i*2+1,mid+1,r);
}
void update(int i,int l,int r,int val)
{
    if(tree[i].r<l||tree[i].l>r)
        return ;
    if(tree[i].l>=l&&tree[i].r<=r)
    {
        tree[i].mx=val;
        tree[i].sum=val;
        return ;
    }
    update(i*2,l,r,val);
    update(i*2+1,l,r,val);
    push_up(i);
}
int query_max(int i,int l,int r)
{
    if(tree[i].l>r||tree[i].r<l)
        return -1e15;        //返回最小值
    if(tree[i].l>=l&&tree[i].r<=r)
        return tree[i].mx;
    return max(query_max(i*2,l,r),query_max(i*2+1,l,r));
}
int query_sum(int i,int l,int r){
    if(tree[i].l>r||tree[i].r<l)
        return 0;
    if(tree[i].l>=l&&tree[i].r<=r)
        return tree[i].sum;
    return query_sum(i*2,l,r)+query_sum(2*i+1,l,r);

}

int dfs1(int now,int d,int pre){
    fa[now]=pre;
    dep[now]=d;
    sz[now]=1;
    int mx=-1;
    for(auto x:vec[now]){
        if(x==pre) continue;
        sz[now]+=dfs1(x,d+1,now);
        if(sz[x]>mx){
            mx=sz[x];
            son[now]=x;
        }
    }
    return sz[now];
}
void dfs2(int now,int pre){
    id[now]=++cnt;
    if(now==son[fa[now]]) top[now]=top[fa[now]];
    else top[now]=now;
    if(son[now]) dfs2(son[now],now); //优先更新重子节点
    for(auto x:vec[now]){
        if(x==pre) continue;
        if(x==son[now]) continue;
        dfs2(x,now);
    }
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);
    int n;cin>>n;
    for(int i=1;i<=n-1;++i){
        int u,v;cin>>u>>v;
        vec[u].push_back(v);
        vec[v].push_back(u);
    }
    dfs1(1,0,0);
    dfs2(1,0);
    build(1,1,n);
    for(int i=1;i<=n;++i){
        cin>>a[i];
        update(1,id[i],id[i],a[i]);
    }
    /*for(int i=1;i<=n;++i)
        cout<<query_max(1,id[i],id[i])<<endl;*/
    int q;cin>>q;
    while(q--){
        string op;cin>>op;
        int u,v;cin>>u>>v;
        if(op=="CHANGE"){
            update(1,id[u],id[u],v);
        }
        else if(op=="QMAX"){
            int mx=-1e15;
            while(top[u]!=top[v]){
                if(dep[top[u]]<dep[top[v]])
                    swap(u,v);
                mx=max(mx,query_max(1,id[top[u]],id[u]));
                u=fa[top[u]];
            }
            if(id[u]>id[v]) swap(u,v);
            mx=max(mx,query_max(1,id[u],id[v]));
            cout<<mx<<endl;
        }
        else{
            int ans=0;
            while(top[u]!=top[v]){
                if(dep[top[u]]<dep[top[v]])
                    swap(u,v);
                ans+=query_sum(1,id[top[u]],id[u]);
                u=fa[top[u]];
            }
            if(id[u]>id[v]) swap(u,v);
            ans+=query_sum(1,id[u],id[v]);
            cout<<ans<<endl;
        }
    }

    return 0;
}
```
#####四边形不等式优化区间dp
```cpp
区间单调性:任意l<l1<r1<r，w(l1,r1)<w(l,r)
四边形不等式 l<l1<r<r1,w(l1,r1)+w(l,r)<=w(l,r1)+w(l1,r)
定理一：若w（l，r）满足区间包含单调性和四边形不等式，则fl,r 满足四边形不等式
定理二：ml,r-1<=ml,r<=ml+1,r
// C++ Version
for (int len = 2; len <= n; ++len)  // 枚举区间长度
  for (int l = 1, r = len; r <= n; ++l, ++r) {
    // 枚举长度为len的所有区间
    f[l][r] = INF;
    for (int k = m[l][r - 1]; k <= m[l + 1][r]; ++k)
      if (f[l][r] > f[l][k] + f[k + 1][r] + w(l, r)) {
        f[l][r] = f[l][k] + f[k + 1][r] + w(l, r);  // 更新状态值
        m[l][r] = k;  // 更新（最小）最优决策点
      }
  }
```
#####差分约束
```cpp
#define PII pair<int,int>
int vis[N+5],dis[N+5],cnt[N+5];
vector<PII> vec[N+5]; 
bool spfa(int s)
{
    memset(dis,inf,sizeof(dis)); 
    memset(vis,0,sizeof(vis));
    memset(cnt,0,sizeof(cnt));
    dis[s]=0;vis[s]=cnt[s]=1;//s为起点
    queue<int> q;
    q.push(s);
    while(q.size())
    {
        int now=q.front();
        q.pop();
        vis[now]=0; //该点不在队列中
        for(int i=0;i<vec[now].size();++i)
        {
            int v=vec[now][i].first;
            int len=vec[now][i].second;
            if(dis[v]>dis[now]+len)
            {
                dis[v]=dis[now]+len;
                if(!vis[v])
                {
                    q.push(v);
                    vis[v]=1;       //v入队
                    if(++cnt[v]>n) //入队次数超过n
                        return false;
                }
            }
        }
    }
    return true;
}
```
#####tarjan求割点
```cpp
#include <bits/stdc++.h>
#define PII pair<int,int>
#define int long long
using namespace std;
const int N=1e5;
const int mod=1e9+7;
vector<int> g[N+5];
int low[N+5],dfsn[N+5],id;
vector<int> ans;
void tarjan(int now,int mx){
    low[now]=dfsn[now]=++id;
    int child=0;
    for(auto x:g[now]){
        if(!dfsn[x]){
            tarjan(x,0);
            low[now]=min(low[now],low[x]);
            child+=(low[x]>=dfsn[now]);
        }
        else
            low[now]=min(low[now],dfsn[x]);
    }
    if(child>mx){
        ans.push_back(now);
    }
}
signed main()
{
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);std::cout.tie(0);
    memset(low,0x3f,sizeof low);
    int n,m;cin>>n>>m;
    for(int i=1;i<=m;++i){
        int u,v;cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    for(int i=1;i<=n;++i){
        if(!dfsn[i]){
            tarjan(i,1);
        }
    }
    cout<<ans.size()<<endl;
    sort(ans.begin(),ans.end());
    for(auto x:ans){
        cout<<x<<" ";
    }
    cout<<endl;
    return 0;
}
```

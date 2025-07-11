# item embeddign 说明
目前的embedding方式为将 'title', 'text', 'average_rating', 'rating_number', 'details', 'categories', 'price', 'description', 'features'简单地embedding后取平均值，缺失值不参与embedding。

这样损失了评论的权威性信息，即 'helpful_vote' 的信息。

筛选完之后还剩 `1,261,420` 个商品数据。

# 亚马逊评论数据集说明
使用 `Electronics` 分类，一共有两个数据文件：评论文件`Electronics.jsonl`和元数据文件`meta_Electronics.jsonl`。

推荐系统的数据存储形式：
1. 产品 id
2. 产品反馈 embedding
3. 产品属性 embedding

Embedding 所需要的信息：
* `Electronics.jsonl`
    * 'title', 'text' 评论的内容，是要被embedding的对象。产品反馈 embedding
    * 'helpful_vote' 可以作为评论的权重，很多人认为有用则说明这个评论很权威。产品反馈 embedding
    * 'parent_asin' 是把 embedding 对产品的指针。
* `meta_Electronics.jsonl`
    * 'average_rating', 'rating_number' 用户的评分和参与评分的人数，可作为推荐系统的参考，人数越多一般说明销量大。评分越高越优先推荐，相同评分下评论数量越多越优先推荐。产品反馈 embedding
    * 'title', 'details', 'categories', 'price' details 是产品详情，包括材质、品牌、尺寸等，没有缺失, categories 是产品的层次类别，是非常基础的信息。产品反馈 embedding
    * 'description', 'features' 单个拎出来都有很多产品缺失，但两个都缺失的不多。description 是对产品的描述，倾向是广告词，features 是 description 的关键词版本，可以作为 detials 的补充，信息量 'categories' < 'details' < 'description' < 'features' 。产品反馈 embedding
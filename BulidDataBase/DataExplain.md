# 亚马逊评论数据集说明
## 文件 `meta_Electronics.jsonl`

```py
# 读取JSONL文件
def load_jsonl(file_path):
    """读取JSONL格式的文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
```
加载数据
```py
# 加载数据
path = 'meta_Electronics.jsonl'
try:
    # 读取JSONL数据
    raw_data = load_jsonl(path)
    
    # 转换为DataFrame以便查看
    df = pd.DataFrame(raw_data)
```
一共有 `1610012` 条数据
表头为
```
>>> df.columns
Index(['main_category', 'title', 'average_rating', 'rating_number', 'features',
       'description', 'price', 'images', 'videos', 'store', 'categories',
       'details', 'parent_asin', 'bought_together'],
      dtype='object')
```

| 字段名 | 类型 | 说明 | 示例| 
|--------|------| ---------| ---|
| main_category | str | 产品的主分类 |All Electronics、Computers、Cell Phones & Accessories|
| title | str | 产品标题 | 'Dock Audio Extender Adapter Converter Cable for iPod iPhone 4 4S iPhone 5 5S iPad, Samsung & Other Smart Phones'|
| average_rating |float | 用户的评分| 3.8 |
| rating_number | int | 参与产品评分的数量 |小到1，多到上万|
| features | list | 产品的功能特点 | '♥Best Gift♥-Catoon shape design different from traditional USB Flash Drive,cute and novelty,cheer up the working day.Cute outside but work well.It will be an amazing gift for your kid,friends and even significant others', 也有很多缺失值 []|
| description | list | 产品描述 |['HDMI In - HDMI Out']，但是 不少产品没有描述，即[]|
| price | float | 价格单位为美元 | 多贵的多有，但也有很多没有价格，NaN |
|images | list | 产品图片。每张图片都有不同的尺寸（缩略图、大图、高分辨率）。“variant”字段显示图片的位置。| [{'thumb': 'https://m.media-amazon.com/images/I/41qrX56lsYL._AC_US40_.jpg', 'large': 'https://m.media-amazon.com/images/I/41qrX56lsYL._AC_.jpg', 'variant': 'MAIN', 'hi_res': None}] |
| videos | list | 产品视频，包括标题和网址。 | 很多都没有 |
| store | str | 产品的店铺名称 | Fat Shark、SIIG   |
| categories | list |产品的层次类别。 |['Electronics', 'Computers & Accessories', 'Laptop Accessories', 'Skins & Decals', 'Decals'] |
|details | dic | 产品详情，包括材质、品牌、尺寸等。|  {'Product Dimensions': '3 x 3 x 1 inches', 'Item Weight': '0.81 ounces', 'Item model number': 'POM-S-001', 'Best Sellers Rank': {'Electronics': 254105, 'Smart Arm & Wristband Accessories': 9928}, 'Is Discontinued By Manufacturer': 'No', 'Other display features': 'Wireless', 'Color': 'Black&Rose', 'Manufacturer': 'NANW', 'Date First Available': 'September 27, 2018'}| 
| parent_asin | str | 产品的父ID | B07848ZT9T 和 B00R6R82HS 等|
| bought_together | list | 来自网站的推荐捆绑包 | 很多都没有，即 None |

## 文件 `Electronics.jsonl`
一共有 `43886944` 条评论数据，包含 `1609860` 件商品。
```py
>>> df['parent_asin'].unique().shape
(1609860,)
```

dict_keys(['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'])
| 字段名 | 类型 | 说明 | 示例| 
|--------|------| ---------| ---|
|rating|float |评分 | (1.0-5.0)|
| title | str |评论标题 | 'Didn’t work at all lenses loose/broken.' |
| text | str |  评论内容 | 并不是一个商品后面跟随着n条评论，而是一条数据一条评论。'These didn’t work. Idk if they were damaged in shipping or what, but the lenses were loose or something. I could see half a lens with its edge in the frame and the rest was missing. It looked like it came loose or was broken.' |
| images |list | 图片 | 是买家评价时附带的图片，也不是每一个用户都提供了，按照分辨率分为三种大小：small、medium、large |
| asin | str | 商品ID | 'B083NRGZMM' 查了几个，asin 和 parent_asin 一致 |
|parent_asin | str |注意：不同颜色、款式和尺寸的产品通常属于同一个父 ID。之前的亚马逊数据集中的“asin”实际上是父 ID。请使用父 ID 查找产品元数据。|
| user_id | str | 评论用户ID |
|timestamp |int | 时间戳 |
| verified_purchase | bool | 是否购买
| helpful_vote | int | 有用投票数 | 
# AI导购系统 AI-Concierge 
## 项目简介
“智能侦探AI导购系统”是一款面向电商平台的智能推荐与导购解决方案。系统聚焦于用户需求表达模糊、冷启动、新用户兴趣捕捉、商品隐性关系难以挖掘等行业痛点，创新性地结合大语言模型（LLM）与高效向量检索技术，实现了主动推理、个性化、多轮对话的智能导购体验，显著提升用户转化率与平台服务能力。

## 业务背景

在电商场景中，用户常面临几大痛点：

1. 需求表达模糊：用户无法准确描述需求（如“适合海边度假的裙子”），导致搜索效率低下；选择困难：平台推荐算法依赖历史行为，缺乏实时交互式引导，用户易陷入信息过载。
2. 现有解决方案（如关键词搜索、静态推荐）无法动态理解用户意图并主动挖掘隐性需求。结合三个项目技术，可构建一个主动推理型AI导购，通过对话交互精准定位用户需求，同时优化平台转化率。
3. 冷启动困难—在缺少用户行为信息时(如：点击、购买等)，传统的深度学习推荐方案依赖历史信息难以有效推荐。
4. （待做）商品之间的显性关系容易被发现而隐性关系则不容易发觉，而往往商品隐性特征的才是决定用户是否购买的关键。例如用户的搜索中透露出了他想要优雅地躲雨，这个时候在推荐雨衣雨伞的同时推荐雨鞋套也是不错的主意。

## 核心工作与成果
### 1. 多轮对话与主动推理
* 支持与用户进行多轮自然语言对话，主动引导用户表达真实需求。
* 结合用户实时行为（点击、加购、收藏、购买等），动态调整推荐策略，实现从被动响应到主动挖掘。

### 2. 大模型微调与意图理解
* 基于Qwen大语言模型，利用问答语义数据和电商品牌知识库，通过LoRA技术进行场景化微调。
* 显著提升模型对电商买家意图、商品属性、场景语境的理解能力。

### 3. 商品数据向量化与高效检索
* 对160万商品数据进行清洗，保留120万高质量商品（清洗率25%）。
* 针对4380万条评论和商品9大属性，生成多维度Embedding，采用通道注意力机制加权聚合，获得高区分度商品特征向量。
* 利用FAISS构建高效向量检索库，支持亿级商品的毫秒级相似度搜索。
* 引入时间特征，提升冷启动商品的召回率9%。

### 4. 需求匹配与解释性推荐
* 整理超1亿条用户行为，提取2610万条高价值行为序列。
* 基于用户行为序列，微调大模型生成用户Query，实现精准需求匹配。
* 推荐结果附带推理路径解释，提升推荐透明度与用户信任。
* 构建用户偏好画像，支持长期个性化服务优化。

### 5. 用户隐性需求捕获
* 构建商品知识图谱，结合多轮对话动态理解用户高阶意图。
* 精准推荐用户未直接表达但潜在需要的商品（如买雨伞时推荐防水鞋套），提升顺手购买转化率。

## 技术亮点
* 大模型微调：结合LoRA与Qwen，深度适配电商场景，理解复杂用户意图。
* 多模态特征融合：评论、属性、时间等多维特征融合，Embedding表达更丰富。
* 高效向量检索：FAISS支撑亿级商品的实时相似度搜索，推荐响应毫秒级。
* 解释性AI推荐：每次推荐均可追溯推理路径，提升用户体验与信任感。
* 冷启动优化：新用户、新商品均可获得高质量推荐，冷启动召回率显著提升。
* 知识图谱驱动：商品知识图谱助力隐性需求捕获，实现更智能的商品联动推荐。

## 应用价值
* 提升转化率：主动推理与多轮交互，精准定位用户需求，显著提升购买转化。
* 增强用户体验：个性化推荐与解释性反馈，增强用户信任与平台粘性。
* 冷启动无忧：新用户、新商品均可获得高质量推荐，助力平台快速扩展。
* 智能商品联动：隐性需求捕获与知识图谱联动，提升顺手购买与客单价。

## 适用场景
1. 捆绑推荐：相关商品组合推荐
2. 探索推荐：帮助用户发现新商品

# To do list
- [x]  商品 SQL 数据库建立
- [x]  微调大模型，理解电商场景的意图表达
- [x]  商品特征 Embedding 生成
- [x]  用户动作 List 生成
- [ ]  基于 Qwen 大模型和用户动作 List 的 Query 生成
- [x]  商品 Faiss 向量库建立
- [x]  商品的向量召回
- [ ]  召回商品排序
- [ ]  基于通道注意力机制的 Embedding 权重模型训练
- [ ]  商品知识图谱构建
- [ ]  模型评估
- [ ]  算法对比
- [ ]  项目前端界面与主页搭建
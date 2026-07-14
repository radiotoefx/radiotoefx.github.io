/*
 * 更新博客最常修改的文件。
 * 新文章：复制 articles 数组中的一项，修改 slug、标题、分类与正文。
 * 图片：放进 images/ 文件夹，然后在正文写 <figure>...</figure>。
 */
window.RADIOTOE_CONTENT = {
  categories: [
    { id: "foundations", name: "基础结构" },
    { id: "training", name: "训练与数据" },
    { id: "inference", name: "推理与效率" },
    { id: "alignment", name: "对齐与智能" }
  ],
  articles: [
    {
      slug: "transformer-from-first-principles",
      title: "从第一性原理理解 Transformer",
      summary: "注意力机制究竟在计算什么？从表示、相似度到信息聚合，重新走过 Transformer 的核心路径。",
      category: "foundations",
      tags: ["Transformer", "Attention"],
      date: "2026.07.14",
      status: "published",
      body: `
        <p class="lead">Transformer 最容易被误解成一组需要背诵的公式：Query、Key、Value，多头注意力，残差连接。可如果只记住这些名词，我们仍然不知道它为什么有效。更好的起点是回到一个朴素问题：<strong>怎样让一句话里的每个位置，都能按需读取其他位置的信息？</strong></p>
        <div class="article-index"><span>本文路径</span><ol><li>序列为什么难</li><li>注意力在计算什么</li><li>多头与残差</li><li>从模块到语言模型</li></ol></div>

        <h2><span>01</span>序列建模真正困难的地方</h2>
        <p>语言是一条序列，但理解语言并不只依赖相邻关系。在“银行把利率下调，因为它担心需求下降”里，“它”指向“银行”；这个联系跨过了若干词。模型需要把远处的信息带到当前位置，并判断哪些信息值得保留。</p>
        <p>早期的循环神经网络按顺序处理 token：读入第一个，再把压缩后的状态传给第二个。它的直觉很自然，却有两个限制。第一，信息必须经过一条很长的路径才能从句首走到句尾；第二，时间步之间存在依赖，难以把整段序列充分并行化。</p>
        <blockquote>Transformer 的关键变化不是“更复杂”，而是把信息传递从一条链，改造成任意位置之间可以直接建立连接的网络。</blockquote>

        <h2><span>02</span>注意力：一次有目的的信息检索</h2>
        <p>假设每个 token 已经被映射为向量。对于当前位置，模型要完成三件事：描述自己正在寻找什么，描述每个位置能够提供什么，再把匹配到的内容取回来。这三种角色分别就是 Query、Key 和 Value。</p>
        <div class="concept-grid">
          <div><b>Q</b><small>Query</small><p>我现在需要寻找什么？</p></div>
          <div><b>K</b><small>Key</small><p>我可以通过什么特征被找到？</p></div>
          <div><b>V</b><small>Value</small><p>如果被选中，我要提供什么信息？</p></div>
        </div>
        <p>Q、K、V 并不是三份独立输入。它们来自同一组 token 表示，只是分别乘上三个可学习矩阵。模型在训练中逐渐学会怎样投影适合“提问”“匹配”与“传递”的方向。</p>
        <div class="formula"><small>Scaled Dot-Product Attention</small><div>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>)V</div></div>
        <ol class="steps">
          <li><strong>计算相似度。</strong>用 Q 与所有 K 做点积，得到当前位置对其他位置的匹配分数。</li>
          <li><strong>控制数值尺度。</strong>除以 √d<sub>k</sub>，避免点积过大使 softmax 饱和。</li>
          <li><strong>转换为权重。</strong>softmax 把分数变成总和为 1 的注意力分布。</li>
          <li><strong>聚合信息。</strong>用这些权重对 V 加权求和，得到当前位置新的表示。</li>
        </ol>
        <aside><small>一个重要细节</small><p>生成式语言模型使用因果掩码：当前位置不能看到未来 token，未来位置的注意力权重会被置为零。</p></aside>

        <h2><span>03</span>为什么需要多头注意力</h2>
        <p>一次注意力只能在一个表示子空间里组织关系。但语言里的关系并不单一：有的头可能追踪指代，有的头偏向局部搭配，有的头关注标点或段落边界。多头注意力让模型同时运行多组独立投影，再把结果拼接起来。</p>
        <figure>
          <img src="./images/transformer-architecture.svg" alt="Transformer 模块中的注意力、残差与前馈网络结构" />
          <figcaption><span>FIG. 01</span>一个 Transformer 模块的信息路径。新增图片时只需放进 images 文件夹并填写图片地址。</figcaption>
        </figure>
        <p>“头”不必各自对应一种人类可命名的语法现象。更准确的理解是：它们给模型提供多个并行的关系通道，让信息可以从不同角度被选择和组合。</p>

        <h2><span>04</span>注意力之外：残差与前馈网络</h2>
        <p>Transformer 层并不只有注意力。注意力负责位置之间的信息交换；前馈网络则对每个位置独立地进行非线性变换。可以把前者理解为“通信”，把后者理解为“思考”。</p>
        <div class="formula compact"><div>x′ = x + Attention(Norm(x))</div><div>y = x′ + MLP(Norm(x′))</div></div>
        <p>残差连接保留一条清晰的信息与梯度通道；归一化帮助各层维持稳定的数值分布。模块被重复堆叠后，模型获得的是一系列逐层更新，而非一次孤立的检索。</p>

        <h2><span>05</span>从 Transformer 到语言模型</h2>
        <p>训练时，文本先被切分为 token，并与位置信息相加。经过多层 Transformer 后，每个位置得到一个包含上下文的向量。最后的线性层把它投影到词表大小，再由 softmax 给出下一个 token 的概率分布。</p>
        <div class="formula"><small>Autoregressive objective</small><div>P(x) = ∏ P(x<sub>t</sub> | x<sub>&lt;t</sub>)</div></div>
        <p>模型并没有直接接受“语法”“事实”或“推理规则”的标签。它只是在大量上下文中不断预测下一个 token。为了把预测做得更好，模型被迫学习词义、句法、世界知识以及某些可复用的计算模式。</p>

        <h2><span>06</span>把公式重新压缩成直觉</h2>
        <ul class="summary-list">
          <li><strong>表示：</strong>把离散 token 放进连续向量空间。</li>
          <li><strong>路由：</strong>用注意力决定信息应从哪里流向哪里。</li>
          <li><strong>变换：</strong>用前馈网络加工每个位置获得的信息。</li>
          <li><strong>迭代：</strong>重复堆叠，让简单关系逐层组合成复杂表示。</li>
        </ul>
        <p>Transformer 的力量不来自某个孤立技巧，而来自一种可扩展的信息组织方式。理解这一点之后，Q、K、V 就不再是三个需要死记的字母，而是同一件事的三个视角——寻找、匹配与传递。</p>
        <div class="article-end"><span>END / 01</span><p>下一篇：Token、向量与语言的入口</p></div>
      `
    },
    {
      slug: "tokenization-and-embedding",
      title: "Token、向量与语言的入口",
      summary: "模型看到的并不是词。理解分词、嵌入空间与位置编码如何把文本变成可计算的对象。",
      category: "foundations", tags: ["Tokenization", "Embedding"], date: "计划中", status: "planned",
      outline: ["文本如何成为数字", "子词分词", "嵌入空间", "位置信息的注入"]
    },
    {
      slug: "pretraining-objective", title: "下一个词，为什么足够？",
      summary: "从最大似然到交叉熵，讨论简单的预测目标如何在大规模数据上催生复杂能力。",
      category: "training", tags: ["Pre-training", "Loss"], date: "计划中", status: "planned",
      outline: ["语言建模目标", "交叉熵损失", "规模与涌现", "目标函数的边界"]
    },
    {
      slug: "kv-cache-explained", title: "KV Cache 与逐 Token 生成",
      summary: "一次回答是怎样逐步生成的？拆解自回归推理、缓存复用以及显存与速度之间的交换。",
      category: "inference", tags: ["Inference", "KV Cache"], date: "计划中", status: "planned",
      outline: ["自回归解码", "重复计算", "KV Cache", "吞吐与延迟"]
    },
    {
      slug: "rlhf-and-preference", title: "从偏好数据到 RLHF",
      summary: "模型如何从预测文本走向遵循意图？梳理监督微调、奖励模型与策略优化。",
      category: "alignment", tags: ["RLHF", "Alignment"], date: "计划中", status: "planned",
      outline: ["预训练模型的局限", "监督微调", "偏好与奖励模型", "策略优化与对齐税"]
    },
    {
      slug: "reasoning-and-test-time-compute", title: "推理能力与测试时计算",
      summary: "当模型获得更多思考时间时，性能为什么会提升？从采样、验证与搜索的角度观察推理。",
      category: "alignment", tags: ["Reasoning", "Test-time Compute"], date: "计划中", status: "planned",
      outline: ["什么是推理", "计算预算", "采样与验证", "搜索和过程监督"]
    }
  ]
};

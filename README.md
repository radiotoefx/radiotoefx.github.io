# radiotoe

`radiotoefx.github.io` 的完整静态博客工程。视觉沿用原版，内容层改为 Astro Content Collections：每篇文章独立存放，首页、分类、目录、全文搜索、上一篇/下一篇和站点地图都在构建时自动生成。

## 本地运行

安装 Node.js 22，在项目根目录执行：

```bash
npm install
npm run dev
```

正式检查与构建：

```bash
npm run check
npm run build
```

## 新增文章

```bash
npm run new:post -- flow-matching "理解 Flow Matching"
```

可在创建时直接指定分类：

```bash
npm run new:post -- flow-matching "理解 Flow Matching" --category foundations
```

系统会生成 `src/content/posts/flow-matching/index.md`。把图片放进同一目录，用相对路径引用：

```markdown
![图的准确说明](./architecture.png)
```

公式直接使用 LaTeX：

```markdown
行内公式 $s(x)=\nabla_x\log p(x)$ 自然进入段落。

$$
q(x_t\mid x_0)=
\mathcal N\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right).
$$
```

新文章默认 `draft: true`；完成后改成 `draft: false`，它会自动进入首页、分类、侧栏、搜索、上下篇和站点地图。

## 删除文章

```bash
npm run remove:post -- flow-matching
```

文章会从站点移出，并在本机 `.trash/` 中保留可恢复副本。直接删除对应文章目录也可以。

## 分类管理与文章移动

分类集中保存在 `src/data/categories.json`，首页筛选、左侧目录、文章页分类名和搜索会自动读取它。日常操作不需要手改这个文件。

查看全部分类：

```bash
npm run category:list
```

新增分类：

```bash
npm run category:add -- diffusion "扩散模型"
```

修改分类显示名称：

```bash
npm run category:rename -- diffusion "扩散与生成"
```

把文章移动到另一个分类：

```bash
npm run post:move -- dit-from-ddpm-to-scalable-transformers diffusion
```

移动写作计划：

```bash
npm run roadmap:move -- tokenization-and-embedding diffusion
```

如果需要修改分类 ID，下面的命令会自动同步所有正式文章与写作计划：

```bash
npm run category:change-id -- diffusion generative-models
```

只允许删除没有文章、也没有写作计划的空分类：

```bash
npm run category:remove -- diffusion
```

这样可以避免误删分类后留下失效条目。命令完成后运行 `npm run build`，全站会自动重新生成。

## 写作计划

编辑 `src/data/roadmap.json`。计划条目与正式文章分离，不会进入搜索、侧栏或上下篇。发布后从计划文件移除；自动检查会阻止同一 slug 同时存在于两处。

## GitHub Pages 部署

把本工程的全部内容放在 `radiotoefx.github.io` 仓库根目录，保留原仓库的 `.git/`。依次执行：

```bash
npm install
npm run build
git add .
git commit -m "Rebuild blog with independent Markdown posts"
git push origin main
```

打开仓库 `Settings → Pages → Build and deployment`，把 Source 设为 `GitHub Actions`。以后推送 `main` 会自动校验、构建并发布，网站地址仍是 `https://radiotoefx.github.io`。

原版能运行，是因为仓库根目录最终提供普通静态文件。新版只增加推送后的自动构建步骤；产物仍是 HTML、CSS、JavaScript 和图片，不需要数据库或长期运行的服务器。

原版的 `/?article=文章-slug` 链接会自动跳转到新版 `/posts/文章-slug/`，
因此以前保存或分享过的文章链接仍然可用。

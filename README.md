# radiotoe

这是 radiotoe 的 GitHub Pages 静态博客源码，目标地址：

`https://radiotoefx.github.io`

## 第一次上线

1. 在 GitHub 新建公开仓库：`radiotoefx.github.io`。
2. 把本项目中的所有文件上传到仓库根目录。
3. 打开仓库 **Settings → Pages**。
4. 在 **Build and deployment** 中选择 **Deploy from a branch**。
5. Branch 选择 **main**，目录选择 **/(root)**，保存。
6. 等待 GitHub Pages 完成部署后访问上面的地址。

## 新增文章

主要内容位于 `content/articles.js`：

1. 复制 `articles` 数组里的一篇文章。
2. 修改 `slug`、`title`、`summary`、`category`、`tags` 和 `date`。
3. 将 `status` 改为 `published`。
4. 在 `body` 中写正文 HTML。
5. 提交修改，GitHub Pages 会自动重新发布。

## 增加图片

1. 把图片放进 `images/`，建议每篇文章建立一个子文件夹。
2. 在正文中加入：

```html
<figure>
  <img src="./images/文章文件夹/图片.jpg" alt="准确描述图片内容" />
  <figcaption><span>FIG. 01</span>图片说明</figcaption>
</figure>
```

图片会自动适配正文宽度、手机屏幕和三套页面主题。

## 修改样式

- 页面样式：`styles.css`
- 页面功能：`app.js`
- 文章与分类：`content/articles.js`
- 技术插图：`images/`

这是纯静态网站，不需要安装依赖或运行构建命令。

import {mkdir,writeFile,access} from "node:fs/promises";
import {resolve} from "node:path";
import {readCategories,requireCategory} from "./content-tools.mjs";
const args=process.argv.slice(2),categoryFlag=args.indexOf("--category");
let categoryId;
if(categoryFlag!==-1){categoryId=args[categoryFlag+1];args.splice(categoryFlag,2)}
const [slug,...parts]=args,title=parts.join(" ").trim();
if(!slug||!title){console.error('用法：npm run new:post -- flow-matching "理解 Flow Matching" --category foundations');process.exit(1)}
if(!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)){console.error("slug 只能包含小写字母、数字和连字符。");process.exit(1)}
const categories=await readCategories();
categoryId=categoryId||categories[0]?.id;
try{requireCategory(categories,categoryId)}catch(error){console.error(error.message);process.exit(1)}
const dir=resolve("src/content/posts",slug),file=resolve(dir,"index.md");
try{await access(dir);console.error(`文章目录已存在：${dir}`);process.exit(1)}catch{}
const today=new Date().toISOString().slice(0,10);
await mkdir(dir,{recursive:true});
await writeFile(file,`---
slug: "${slug}"
title: "${title.replaceAll('"','\\"')}"
description: "在这里写一行准确的文章摘要。"
date: ${today}
category: ${categoryId}
tags: []
draft: true
---

在这里开始正文。草稿不会出现在正式网站中。

## 第一个问题

先定义问题，再给出推导与证据。

$$
f(x)=x
$$
`,"utf8");
console.log(`已创建：${file}`);

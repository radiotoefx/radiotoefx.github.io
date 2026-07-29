import {readFile,readdir} from "node:fs/promises";import{resolve}from"node:path";
const root=resolve("src/content/posts"),dirs=(await readdir(root,{withFileTypes:true})).filter(e=>e.isDirectory()).map(e=>e.name);
const required=["slug","title","description","date","category","draft"],slugs=new Map(),errors=[];
const categories=JSON.parse(await readFile(resolve("src/data/categories.json"),"utf8"));
const valid=new Set();
if(!Array.isArray(categories)||!categories.length)errors.push("categories.json: 至少需要一个分类");
else for(const [index,item] of categories.entries()){if(!item||typeof item!=="object"){errors.push(`categories.json: 第 ${index+1} 项格式无效`);continue}
if(!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(item.id||""))errors.push(`categories.json: 分类 ID 无效 ${item.id??""}`);
if(!String(item.name||"").trim())errors.push(`categories.json: 分类 ${item.id??index+1} 缺少名称`);
if(valid.has(item.id))errors.push(`categories.json: 分类 ID 重复 ${item.id}`);valid.add(item.id)}
for(const dir of dirs){let source;try{source=await readFile(resolve(root,dir,"index.md"),"utf8")}catch{errors.push(`${dir}: 缺少 index.md`);continue}
if(/\$\$[ \t]+\$\$/.test(source))errors.push(`${dir}: 两个公式分隔符不能写在同一行`);
const match=source.match(/^---\n([\s\S]*?)\n---/);if(!match){errors.push(`${dir}: 缺少 frontmatter`);continue}
const values=Object.fromEntries(match[1].split("\n").flatMap(line=>{const p=line.match(/^([A-Za-z][\w-]*):\s*(.*)$/);return p?[[p[1],p[2].replace(/^["']|["']$/g,"")]]:[]}));
for(const key of required)if(!values[key])errors.push(`${dir}: 缺少 ${key}`);if(values.slug!==dir)errors.push(`${dir}: slug 必须与目录名一致`);
if(slugs.has(values.slug))errors.push(`${dir}: slug 重复`);else slugs.set(values.slug,dir);if(values.category&&!valid.has(values.category))errors.push(`${dir}: 未知分类 ${values.category}`)}
const roadmap=JSON.parse(await readFile(resolve("src/data/roadmap.json"),"utf8"));for(const item of roadmap){if(slugs.has(item.slug))errors.push(`roadmap: ${item.slug} 已是正式文章`);if(!valid.has(item.category))errors.push(`roadmap: ${item.slug} 分类无效`)}
if(errors.length){console.error(`内容检查失败：\n- ${errors.join("\n- ")}`);process.exit(1)}console.log(`内容检查通过：${dirs.length} 篇文章，${roadmap.length} 个写作计划，${categories.length} 个分类。`);

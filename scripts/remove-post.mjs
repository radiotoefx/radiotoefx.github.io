import {access,mkdir,rename} from "node:fs/promises";
import {resolve} from "node:path";
const [slug]=process.argv.slice(2);
if(!slug||!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)){console.error("用法：npm run remove:post -- flow-matching");process.exit(1)}
const source=resolve("src/content/posts",slug),trash=resolve(".trash",`${slug}-${new Date().toISOString().replaceAll(":","-").replaceAll(".","-")}`);
try{await access(source)}catch{console.error(`没有找到文章：${source}`);process.exit(1)}
await mkdir(resolve(".trash"),{recursive:true});await rename(source,trash);console.log(`文章已移出站点，并保留在：${trash}`);

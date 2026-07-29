import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import {
  categoriesFile,
  frontmatterValue,
  postsRoot,
  readCategories,
  readJson,
  roadmapFile,
  slugPattern,
  writeAtomic,
  writeJson
} from "./content-tools.mjs";

const [operation, ...args] = process.argv.slice(2);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function validateId(id) {
  if (!id || !slugPattern.test(id)) {
    fail("分类 ID 只能包含小写字母、数字和连字符。");
  }
}

async function categoryUsage(id) {
  const posts = [];
  const entries = await readdir(postsRoot, { withFileTypes: true });
  for (const entry of entries.filter((item) => item.isDirectory())) {
    const source = await readFile(resolve(postsRoot, entry.name, "index.md"), "utf8");
    if (frontmatterValue(source, "category", `文章 ${entry.name}`) === id) {
      posts.push(entry.name);
    }
  }
  const roadmap = (await readJson(roadmapFile))
    .filter((item) => item.category === id)
    .map((item) => item.slug);
  return { posts, roadmap };
}

const categories = await readCategories();

if (operation === "list") {
  for (const [index, category] of categories.entries()) {
    console.log(`${index + 1}. ${category.id} — ${category.name}`);
  }
} else if (operation === "add") {
  const [id, ...nameParts] = args;
  const name = nameParts.join(" ").trim();
  validateId(id);
  if (!name) fail('用法：npm run category:add -- diffusion "扩散模型"');
  if (categories.some((item) => item.id === id)) fail(`分类 ${id} 已存在。`);
  if (categories.some((item) => item.name === name)) fail(`分类名称“${name}”已存在。`);
  categories.push({ id, name });
  await writeJson(categoriesFile, categories);
  console.log(`已新增分类：${id}（${name}）`);
} else if (operation === "rename") {
  const [id, ...nameParts] = args;
  const name = nameParts.join(" ").trim();
  validateId(id);
  if (!name) fail('用法：npm run category:rename -- foundations "基础与表示"');
  const category = categories.find((item) => item.id === id);
  if (!category) fail(`分类 ${id} 不存在。`);
  if (categories.some((item) => item.id !== id && item.name === name)) {
    fail(`分类名称“${name}”已存在。`);
  }
  const previous = category.name;
  category.name = name;
  await writeJson(categoriesFile, categories);
  console.log(`已重命名分类：${previous} → ${name}`);
} else if (operation === "change-id") {
  const [oldId, newId] = args;
  validateId(oldId);
  validateId(newId);
  const category = categories.find((item) => item.id === oldId);
  if (!category) fail(`分类 ${oldId} 不存在。`);
  if (categories.some((item) => item.id === newId)) fail(`分类 ${newId} 已存在。`);

  const entries = await readdir(postsRoot, { withFileTypes: true });
  const postWrites = [];
  for (const entry of entries.filter((item) => item.isDirectory())) {
    const file = resolve(postsRoot, entry.name, "index.md");
    const source = await readFile(file, "utf8");
    const frontmatter = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)?.[1];
    if (!frontmatter) fail(`文章 ${entry.name} 缺少 frontmatter。`);
    const updated = frontmatter.replace(
      new RegExp(`^category:\\s*${oldId}\\s*$`, "m"),
      `category: ${newId}`
    );
    if (updated !== frontmatter) postWrites.push([file, source.replace(frontmatter, updated)]);
  }

  const roadmap = await readJson(roadmapFile);
  let roadmapChanged = false;
  for (const item of roadmap) {
    if (item.category === oldId) {
      item.category = newId;
      roadmapChanged = true;
    }
  }

  category.id = newId;
  for (const [file, source] of postWrites) await writeAtomic(file, source);
  if (roadmapChanged) await writeJson(roadmapFile, roadmap);
  await writeJson(categoriesFile, categories);
  console.log(
    `已修改分类 ID：${oldId} → ${newId}；同步 ${postWrites.length} 篇文章` +
    `${roadmapChanged ? "及写作计划" : ""}。`
  );
} else if (operation === "remove") {
  const [id] = args;
  validateId(id);
  const index = categories.findIndex((item) => item.id === id);
  if (index === -1) fail(`分类 ${id} 不存在。`);
  const usage = await categoryUsage(id);
  if (usage.posts.length || usage.roadmap.length) {
    fail(
      `分类 ${id} 仍在使用，不能删除。\n` +
      `正式文章：${usage.posts.join("、") || "无"}\n` +
      `写作计划：${usage.roadmap.join("、") || "无"}\n` +
      "请先用 post:move 或 roadmap:move 移走这些条目。"
    );
  }
  categories.splice(index, 1);
  if (!categories.length) fail("至少需要保留一个分类。");
  await writeJson(categoriesFile, categories);
  console.log(`已删除空分类：${id}`);
} else {
  fail(
    "可用命令：category:list、category:add、category:rename、" +
    "category:change-id、category:remove"
  );
}

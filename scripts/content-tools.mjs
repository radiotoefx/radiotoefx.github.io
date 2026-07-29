import { readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

export const projectRoot = resolve(import.meta.dirname, "..");
export const postsRoot = resolve(projectRoot, "src/content/posts");
export const categoriesFile = resolve(projectRoot, "src/data/categories.json");
export const roadmapFile = resolve(projectRoot, "src/data/roadmap.json");
export const slugPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

export async function writeJson(path, value) {
  await writeAtomic(path, `${JSON.stringify(value, null, 2)}\n`);
}

export async function writeAtomic(path, source) {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, source, "utf8");
  await rename(temporary, path);
}

export function getFrontmatter(source, label = "文章") {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) throw new Error(`${label}缺少 frontmatter。`);
  return match[1];
}

export function frontmatterValue(source, key, label = "文章") {
  const frontmatter = getFrontmatter(source, label);
  const match = frontmatter.match(new RegExp(`^${key}:\\s*(.*?)\\s*$`, "m"));
  return match?.[1].replace(/^["']|["']$/g, "") ?? "";
}

export function replaceFrontmatterValue(source, key, value, label = "文章") {
  const frontmatter = getFrontmatter(source, label);
  const pattern = new RegExp(`^${key}:\\s*.*$`, "m");
  if (!pattern.test(frontmatter)) throw new Error(`${label}缺少 ${key}。`);
  const updated = frontmatter.replace(pattern, `${key}: ${value}`);
  return source.replace(frontmatter, updated);
}

export async function readCategories() {
  const categories = await readJson(categoriesFile);
  if (!Array.isArray(categories)) throw new Error("categories.json 必须是数组。");
  return categories;
}

export function requireCategory(categories, id) {
  const category = categories.find((item) => item.id === id);
  if (!category) {
    const choices = categories.map((item) => `${item.id}（${item.name}）`).join("、");
    throw new Error(`未知分类 ${id}。可用分类：${choices}`);
  }
  return category;
}

export async function updatePostCategory(slug, categoryId) {
  if (!slugPattern.test(slug)) throw new Error(`无效文章 slug：${slug}`);
  const file = resolve(postsRoot, slug, "index.md");
  const source = await readFile(file, "utf8");
  const current = frontmatterValue(source, "category", `文章 ${slug}`);
  if (current === categoryId) return { file, current, changed: false };
  await writeAtomic(
    file,
    replaceFrontmatterValue(source, "category", categoryId, `文章 ${slug}`)
  );
  return { file, current, changed: true };
}

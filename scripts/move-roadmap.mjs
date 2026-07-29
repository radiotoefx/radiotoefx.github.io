import {
  readCategories,
  readJson,
  requireCategory,
  roadmapFile,
  slugPattern,
  writeJson
} from "./content-tools.mjs";

const [slug, categoryId] = process.argv.slice(2);
if (!slug || !categoryId || !slugPattern.test(slug)) {
  console.error("用法：npm run roadmap:move -- 条目-slug 目标分类-id");
  process.exit(1);
}

try {
  const categories = await readCategories();
  const category = requireCategory(categories, categoryId);
  const roadmap = await readJson(roadmapFile);
  const item = roadmap.find((entry) => entry.slug === slug);
  if (!item) throw new Error(`没有找到写作计划：${slug}`);
  const previous = item.category;
  item.category = categoryId;
  await writeJson(roadmapFile, roadmap);
  console.log(
    previous === categoryId
      ? `写作计划 ${slug} 已经位于 ${categoryId}（${category.name}）。`
      : `已移动写作计划：${slug}，${previous} → ${categoryId}（${category.name}）`
  );
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}

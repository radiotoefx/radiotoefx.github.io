import { readCategories, requireCategory, updatePostCategory } from "./content-tools.mjs";

const [slug, categoryId] = process.argv.slice(2);
if (!slug || !categoryId) {
  console.error("用法：npm run post:move -- 文章-slug 目标分类-id");
  process.exit(1);
}

try {
  const categories = await readCategories();
  const category = requireCategory(categories, categoryId);
  const result = await updatePostCategory(slug, categoryId);
  console.log(
    result.changed
      ? `已移动文章：${slug}，${result.current} → ${category.id}（${category.name}）`
      : `文章 ${slug} 已经位于 ${category.id}（${category.name}）。`
  );
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}

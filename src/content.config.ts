import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import categories from "./data/categories.json";

const categoryIds = categories.map((category) => category.id);

const posts = defineCollection({
  loader: glob({ pattern: "**/index.md", base: "./src/content/posts" }),
  schema: z.object({
    slug: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
    title: z.string().min(1),
    description: z.string().min(1),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    category: z.string().refine(
      (value) => categoryIds.includes(value),
      "category 必须存在于 src/data/categories.json"
    ),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false)
  })
});
export const collections = { posts };

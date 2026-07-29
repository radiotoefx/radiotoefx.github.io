import { getCollection, type CollectionEntry } from "astro:content";
export type Post = CollectionEntry<"posts">;
export async function getPublishedPosts() {
  return (await getCollection("posts"))
    .filter((post) => !post.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" })
    .format(date).replaceAll("/", ".");
}
export function withBase(path = "/") {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}` || "/";
}
export const postUrl = (post: Post) => withBase(`/posts/${post.data.slug}/`);

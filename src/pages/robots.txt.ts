import type { APIRoute } from "astro";
import { site } from "../data/site";
export const GET:APIRoute=()=>new Response(`User-agent: *\nAllow: /\nSitemap: ${site.url}/sitemap-index.xml\n`,{headers:{"Content-Type":"text/plain; charset=utf-8"}});

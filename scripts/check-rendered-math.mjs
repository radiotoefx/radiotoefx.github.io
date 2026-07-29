import { readFile, readdir, access } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";

const root = resolve("dist");
const sourceRoot = resolve("src/content/posts");
const publicBase = (process.env.PUBLIC_BASE || "/").replace(/\/+$/, "");
const errors = [];

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(entries.map(async (entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? walk(path) : path;
  }));
  return files.flat();
}

const outputFiles = await walk(root);
const htmlFiles = outputFiles.filter((file) => file.endsWith(".html"));
const cssFiles = outputFiles.filter((file) => file.endsWith(".css"));
const sourceFiles = (await walk(sourceRoot)).filter((file) => file.endsWith(".md"));

let expectedDisplays = 0;
for (const file of sourceFiles) {
  const source = await readFile(file, "utf8");
  const delimiters = source.match(/^\$\$\s*$/gm)?.length ?? 0;
  if (delimiters % 2 !== 0) {
    errors.push(`${relative(sourceRoot, file)}: display-math delimiters are unbalanced`);
  }
  expectedDisplays += delimiters / 2;
}

let renderedDisplays = 0;
for (const file of htmlFiles) {
  const html = await readFile(file, "utf8");
  renderedDisplays += html.match(/class="katex-display"/g)?.length ?? 0;

  const visibleMarkup = html
    .replace(/<script\b[\s\S]*?<\/script>/gi, "")
    .replace(/<annotation\b[\s\S]*?<\/annotation>/gi, "");

  if (visibleMarkup.includes("katex-error")) {
    errors.push(`${relative(root, file)}: contains a KaTeX error node`);
  }
  if (/^\$\$\s*$/m.test(visibleMarkup)) {
    errors.push(`${relative(root, file)}: contains an unrendered display delimiter`);
  }
  if (/\\begin\{(?:aligned|gathered|substack)\}/.test(visibleMarkup)) {
    errors.push(`${relative(root, file)}: contains a visible raw LaTeX environment`);
  }
}

if (renderedDisplays !== expectedDisplays) {
  errors.push(`display-math count mismatch: expected ${expectedDisplays}, rendered ${renderedDisplays}`);
}

const css = (await Promise.all(cssFiles.map((file) => readFile(file, "utf8")))).join("\n");
if (!css.includes(".article-view .article-body .katex-display")) {
  errors.push("built CSS is missing the article display-math rule");
}
if (css.includes("min-width:max-content")) {
  errors.push("built CSS contains the layout-breaking min-width:max-content rule");
}
if (/\.article-view \.article-body \.katex\{[^}]*line-height/.test(css)) {
  errors.push("built CSS overrides KaTeX line-height");
}

for (const cssFile of cssFiles) {
  const source = await readFile(cssFile, "utf8");
  const urls = [...source.matchAll(/url\(([^)]+KaTeX[^)]+)\)/g)]
    .map((match) => match[1].replace(/^['"]|['"]$/g, ""));
  for (const url of urls) {
    const pathname =
      publicBase && url.startsWith(`${publicBase}/`)
        ? url.slice(publicBase.length)
        : url;
    const fontPath = pathname.startsWith("/")
      ? resolve(root, pathname.slice(1))
      : resolve(dirname(cssFile), url);
    try {
      await access(fontPath);
    } catch {
      errors.push(`${relative(root, cssFile)}: missing KaTeX font ${url}`);
    }
  }
}

if (errors.length) {
  console.error(`Rendered math check failed:\n- ${errors.join("\n- ")}`);
  process.exit(1);
}

console.log(`Rendered math check passed: ${renderedDisplays} display equations and all KaTeX fonts are present.`);

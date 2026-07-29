import categoryItems from "./categories.json";

export const site = {
  name: "radiotoe",
  title: "radiotoe — 生成模型与大模型原理笔记",
  description: "从底层理解生成模型与大模型：保留推导、实验与工程实现之间的联系。",
  url: "https://radiotoefx.github.io"
};

export const categories = categoryItems;

export function categoryName(id: string) {
  return categories.find((item) => item.id === id)?.name ?? id;
}

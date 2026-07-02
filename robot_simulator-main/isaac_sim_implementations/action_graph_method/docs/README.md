# 文档索引

`docs/` 是当前目录的正式文档区。这里的文档重点不是“删过什么”，而是回答两个问题：

- 现在这些文件分别负责什么
- 日常应该优先看哪些文件

## 文档入口

### `指令文件.md`

- 面向操作。
- 记录当前主线启动、清场、桥接、验收和交互命令。

### `ACTION_GRAPH_MAINLINE_TECHNICAL_GUIDE.md`

- 面向维护。
- 解释默认主线架构、关键问题、修复手段和调试顺序。

### `PROJECT_STRUCTURE.md`

- 面向整理。
- 按“默认主线 / 运行支撑 / 诊断验收 / 参考保留”解释当前目录。

## 目录外配套文档

- `../README.md`
  - 目录总导航。
- `../diagnostics/README.md`
  - 诊断脚本分类说明。
- `../指令文件.md`
  - 兼容入口，实际指向 `docs/指令文件.md`。

## 文档维护规则

- 改命令：更新 `docs/指令文件.md`
- 改架构、链路、问题复盘：更新 `docs/ACTION_GRAPH_MAINLINE_TECHNICAL_GUIDE.md`
- 改文件分类、入口边界、参数文件归属：同时更新 `README.md` 和 `docs/PROJECT_STRUCTURE.md`
- 改诊断脚本用途：更新 `diagnostics/README.md`

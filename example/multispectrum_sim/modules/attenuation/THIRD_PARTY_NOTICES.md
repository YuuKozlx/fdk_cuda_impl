# 第三方依赖说明 / Third-party notices

## CLI11

- 项目：CLI11
- 版本：2.6.2
- 上游：https://github.com/CLIUtils/CLI11
- 许可证：BSD-3-Clause
- 单头文件：`third_party/CLI11.hpp`
- 许可证正文：`third_party/CLI11-LICENSE.txt`
- 上游发布：https://github.com/CLIUtils/CLI11/releases/tag/v2.6.2
- `CLI11.hpp` SHA-256：
  `227A16FE5F9F8ADA80C3C409492475536F597E7BD83A6C26EACC3C8C149A9295`

CLI11 用于 `xcom_cli` 的命令行参数解析、校验和帮助信息生成。它是仅头文件
依赖，不改变 XCOM 数值核心，也不会给运行时增加额外动态库依赖。

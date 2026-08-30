#pragma once

namespace YK::Helical::Iterative {

// Heli 表示螺旋扫描域，而不是某一种重建算法。解析重建由 WFBP
// 管线负责；这里统一列出可复用逐视图螺旋 geometry 的迭代方法。
enum class EMethod {
    Sirt,
    Sart,
    Ossart,
    Cgls,
    Pwls
};

} // namespace YK::Helical::Iterative

#pragma once
#include <functional>
namespace yk::spectral {
struct TaskProgress {const char* phase="Preparing";int completed=0;int total=0;};
using TaskProgressCallback=std::function<void(TaskProgress)>;
}

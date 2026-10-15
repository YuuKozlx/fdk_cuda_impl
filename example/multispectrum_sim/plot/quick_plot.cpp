// A small data-driven front end to the existing Gnuplot backend.
// No simulation, VTK, Python, or caller-written plotting code is needed.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include "quick_plot_runtime.hpp"
namespace fs = std::filesystem;
using J = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;
struct Table {
    std::vector<std::string> header;
    Matrix rows;
};
std::string quote(const std::string& text)
{
    std::string result = "\"";
    for (char c : text) {
        if (c == '\\' || c == '\"')
            result += '\\';
        if (c == '\n' || c == '\r')
            result += ' ';
        else
            result += c;
    }
    return result + '\"';
}
std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                cell += '"';
                ++i;
            } else
                quoted = !quoted;
        } else if (c == ',' && !quoted) {
            cells.push_back(cell);
            cell.clear();
        } else if (c != '\r')
            cell += c;
    }
    if (quoted)
        throw std::runtime_error("Multiline/unclosed CSV quotes are unsupported");
    cells.push_back(cell);
    return cells;
}
Table readCsv(const fs::path& path, bool header)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read CSV: " + path.u8string());
    Table result;
    std::string line;
    size_t number = 0, columns = 0;
    while (std::getline(in, line)) {
        ++number;
        if (number == 1 && line.rfind("\xef\xbb\xbf", 0) == 0)
            line.erase(0, 3);
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;
        const auto cells = split(line);
        if (header && result.header.empty()) {
            result.header = cells;
            columns = cells.size();
            continue;
        }
        if (!columns)
            columns = cells.size();
        if (cells.size() != columns)
            throw std::runtime_error("CSV column count mismatch at line " + std::to_string(number));
        std::vector<double> row;
        for (const auto& c : cells) {
            size_t n = 0;
            double value;
            try {
                value = std::stod(c, &n);
            } catch (...) {
                throw std::runtime_error("Non-numeric CSV cell at line " + std::to_string(number));
            }
            if (!std::isfinite(value) || c.find_first_not_of(" \t", n) != std::string::npos)
                throw std::runtime_error("Invalid numeric CSV cell at line " + std::to_string(number));
            row.push_back(value);
        }
        result.rows.push_back(std::move(row));
    }
    if (result.rows.empty())
        throw std::runtime_error("CSV contains no data: " + path.u8string());
    return result;
}
size_t column(const Table& t, const J& key)
{
    size_t i = 0;
    if (key.is_number_unsigned() || key.is_number_integer()) {
        const int n = key.get<int>();
        if (n < 0)
            throw std::runtime_error("Column index must be nonnegative");
        i = size_t(n);
    } else {
        const auto name = key.get<std::string>();
        const auto found = std::find(t.header.begin(), t.header.end(), name);
        if (found == t.header.end())
            throw std::runtime_error("CSV column not found: " + name);
        i = found - t.header.begin();
    }
    if (i >= t.rows[0].size())
        throw std::runtime_error("Column index outside CSV");
    return i;
}
fs::path resolve(const fs::path& root, const std::string& name) { return fs::absolute(root / fs::u8path(name)); }
void write(const fs::path& file, const std::string& content)
{
    std::ofstream out(file, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << content;
    out.close();
}
void range(std::ostream& gp, const J& panel, const char* key, const char* axis)
{
    if (panel.contains(key)) {
        const auto r = panel.at(key).get<std::vector<double>>();
        if (r.size() != 2 || !std::isfinite(r[0]) || !std::isfinite(r[1]) || r[0] >= r[1])
            throw std::runtime_error(std::string("Invalid ") + key);
        gp << "set " << axis << "range [" << r[0] << ":" << r[1] << "]\n";
    }
}
std::string panelScript(const J& panel, const fs::path& root, int index)
{
    std::ostringstream gp;
    gp << std::setprecision(17);
    const auto type = panel.value("type", std::string("line"));
    gp << "reset\nset encoding utf8\nset border lc rgb '#555555'\nset grid lc rgb '#dddddd'\nset key opaque\n";
    gp << "set title " << quote(panel.value("title", type)) << "\nset xlabel " << quote(panel.value("xlabel", std::string("X"))) << "\nset ylabel " << quote(panel.value("ylabel", std::string("Y"))) << "\nset zlabel " << quote(panel.value("zlabel", std::string("Z"))) << "\n";
    range(gp, panel, "xlim", "x");
    range(gp, panel, "ylim", "y");
    range(gp, panel, "zlim", "z");
    range(gp, panel, "clim", "cb");
    if (panel.value("xlog", false))
        gp << "set logscale x\n";
    if (panel.value("ylog", false))
        gp << "set logscale y\n";
    gp << "set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21918c', 0.75 '#5ec962', 1 '#fde725')\n";
    if (type == "heatmap" || type == "contour" || type == "surface") {
        const auto t = readCsv(resolve(root, panel.at("file").get<std::string>()), false);
        const auto& z = t.rows;
        if (z.size() < 2 || z[0].size() < 2)
            throw std::runtime_error("Matrix needs at least 2 rows and 2 columns");
        const double x0 = panel.value("x0", 0.), y0 = panel.value("y0", 0.), dx = panel.value("dx", 1.), dy = panel.value("dy", 1.);
        if (!(std::isfinite(dx) && std::isfinite(dy) && dx > 0 && dy > 0 && std::isfinite(x0) && std::isfinite(y0)))
            throw std::runtime_error("Invalid matrix coordinates");
        gp << "$matrix" << index << " << EOD\n";
        for (size_t y = 0; y < z.size(); ++y) {
            for (size_t x = 0; x < z[y].size(); ++x)
                gp << x0 + x * dx << ' ' << y0 + y * dy << ' ' << z[y][x] << '\n';
            gp << '\n';
        }
        gp << "EOD\n";
        gp << "unset key\n";
        if (type == "surface") {
            const auto view = panel.value("view", std::vector<double> { 60, 30 });
            if (view.size() != 2 || !std::isfinite(view[0]) || !std::isfinite(view[1]))
                throw std::runtime_error("view needs elevation, azimuth");
            gp << "set view " << view[0] << ',' << view[1] << "\nset pm3d depthorder\nsplot $matrix" << index << " using 1:2:3 with pm3d\n";
        } else if (type == "contour") {
            const int levels = panel.value("levels", 15);
            if (levels < 2 || levels > 100)
                throw std::runtime_error("levels must be 2..100");
            gp << "set view map\nunset surface\nset contour base\nset cntrparam levels auto " << levels << "\nsplot $matrix" << index << " using 1:2:3 with lines lc palette\n";
        } else
            gp << "plot $matrix" << index << " using 1:2:3 with image\n";
    } else if (type == "line" || type == "scatter" || type == "scatter3d") {
        J series = panel.contains("series") ? panel.at("series") : J::array({ panel });
        if (!series.is_array() || series.empty())
            throw std::runtime_error("series must be a nonempty array");
        std::vector<std::string> terms;
        int k = 0;
        for (const auto& s : series) {
            const auto t = readCsv(resolve(root, s.at("file").get<std::string>()), s.value("header", true));
            const auto xi = column(t, s.value("x", J(0))), yi = column(t, s.value("y", J(1)));
            const size_t zi = type == "scatter3d" ? column(t, s.value("z", J(2))) : 0;
            const std::string name = "$data" + std::to_string(index) + "_" + std::to_string(k++);
            std::vector<std::pair<size_t, double>> filters;
            if (s.contains("where"))
                for (const auto& item : s.at("where").items()) {
                    const double v = item.value().get<double>();
                    if (!std::isfinite(v))
                        throw std::runtime_error("Invalid row filter");
                    filters.emplace_back(column(t, item.key()), v);
                }
            size_t kept = 0;
            gp << name << " << EOD\n";
            for (const auto& row : t.rows) {
                bool matches = true;
                for (const auto& filter : filters)
                    if (std::abs(row[filter.first] - filter.second) > 1e-9 * std::max(1., std::abs(filter.second)))
                        matches = false;
                if (!matches)
                    continue;
                ++kept;
                gp << row[xi] << ' ' << row[yi];
                if (type == "scatter3d")
                    gp << ' ' << row[zi];
                gp << '\n';
            }
            if (!kept)
                throw std::runtime_error("Row filter selected no data");
            gp << "EOD\n";
            std::string style = s.value("style", type == "line" ? std::string("lines") : std::string("points"));
            if (style != "lines" && style != "points" && style != "linespoints")
                throw std::runtime_error("style must be lines, points or linespoints");
            const double width = s.value("width", 2.);
            if (!std::isfinite(width) || width <= 0 || width > 20)
                throw std::runtime_error("Invalid line width");
            std::string term = name + " using " + (type == "scatter3d" ? "1:2:3" : "1:2") + " with " + style + " lw " + std::to_string(width);
            if (s.contains("color"))
                term += " lc rgb " + quote(s.at("color").get<std::string>());
            const int dash = s.value("dash", 1);
            if (dash < 1 || dash > 8)
                throw std::runtime_error("dash must be 1..8");
            term += " dt " + std::to_string(dash);
            term += " title " + quote(s.value("name", std::string("Series ") + std::to_string(k)));
            terms.push_back(term);
        }
        if (type == "scatter3d")
            gp << "set view 60,30\n";
        gp << (type == "scatter3d" ? "splot " : "plot ");
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i)
                gp << ", ";
            gp << terms[i];
        }
        gp << '\n';
    } else
        throw std::runtime_error("Unknown plot type: " + type);
    return gp.str();
}
fs::path executableDirectory()
{
#ifdef _WIN32
    std::wstring name(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, name.data(), DWORD(name.size()));
    if (!n || n == name.size())
        throw std::runtime_error("Cannot locate executable");
    name.resize(n);
    return fs::path(name).parent_path();
#else
    return fs::current_path();
#endif
}
fs::path backend()
{
    const auto local = executableDirectory() / "gnuplot/bin/gnuplot.exe";
    if (fs::exists(local))
        return local;
    const auto configured = fs::u8path(QUICK_PLOT_GNUPLOT);
    if (!fs::exists(configured))
        throw std::runtime_error("Gnuplot runtime missing. Keep the gnuplot folder beside quick_plot.exe.");
    return configured;
}
void launch(const fs::path& exe, const fs::path& script, bool wait)
{
#ifdef _WIN32
    std::wstring command = L"\"" + exe.wstring() + L"\" \"" + script.wstring() + L"\"";
    STARTUPINFOW start {};
    start.cb = sizeof(start);
    PROCESS_INFORMATION process {};
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, exe.parent_path().c_str(), &start, &process))
        throw std::runtime_error("Cannot start Gnuplot: " + std::to_string(GetLastError()));
    CloseHandle(process.hThread);
    if (wait) {
        const auto status = WaitForSingleObject(process.hProcess, 60000);
        DWORD code = 1;
        if (status == WAIT_TIMEOUT) {
            TerminateProcess(process.hProcess, 1);
            CloseHandle(process.hProcess);
            throw std::runtime_error("Plot export exceeded 60 seconds");
        }
        GetExitCodeProcess(process.hProcess, &code);
        CloseHandle(process.hProcess);
        if (code)
            throw std::runtime_error("Gnuplot failed; inspect generated .gp file, exit " + std::to_string(code));
    } else {
        // Keep the front end alive until the interactive window closes. Hosts may
        // terminate detached descendants when the initiating command exits.
        std::cout << "Interactive window: close it to finish. 3D: left drag rotates, middle drag scales.\n" << std::flush;
        WaitForSingleObject(process.hProcess, INFINITE);
        DWORD code=1;GetExitCodeProcess(process.hProcess,&code);
        CloseHandle(process.hProcess);
        if(code)throw std::runtime_error("Interactive Gnuplot failed, exit "+std::to_string(code));
    }
#else
    throw std::runtime_error("This launcher currently targets Windows");
#endif
}
void render(J config, const fs::path& root, bool forceShow, int selected)
{
    const auto panels = config.at("panels");
    if (!panels.is_array() || panels.empty())
        throw std::runtime_error("panels must be a nonempty array");
    const int rows = config.value("rows", 1), cols = config.value("cols", int(panels.size()));
    if (rows < 1 || cols < 1 || rows > 8 || cols > 8 || panels.size() > size_t(rows * cols))
        throw std::runtime_error("Layout must fit panels, at most 8 x 8");
    const int width = config.value("width", 1200), height = config.value("height", 800), font = config.value("font_size", 12);
    if (width < 200 || height < 200 || width > 10000 || height > 10000 || font < 6 || font > 72)
        throw std::runtime_error("Invalid figure/font size");
    if (selected > int(panels.size()) || selected < 0)
        throw std::runtime_error("Panel number is 1-based and outside layout");
    std::ostringstream body;
    if (!selected && panels.size() > 1)
        body << "set multiplot layout " << rows << ',' << cols << " title " << quote(config.value("title", std::string(""))) << "\n";
    for (size_t i = 0; i < panels.size(); ++i)
        if (!selected || int(i + 1) == selected)
            body << panelScript(panels[i], root, int(i));
    if (!selected && panels.size() > 1)
        body << "unset multiplot\n";
    const auto exe = backend();
    const std::string settings = " size " + std::to_string(width) + "," + std::to_string(height) + " font " + quote("Microsoft YaHei," + std::to_string(font)) + " noenhanced\n";
    const auto outputName = config.value("output", std::string(""));
    if (outputName.empty() && !forceShow && !selected && !config.value("show", true))
        throw std::runtime_error("Specify an output file or enable show");
    if (!outputName.empty() && !selected) {
        const auto output = resolve(root, outputName);
        if (output.extension() != ".png" && output.extension() != ".svg")
            throw std::runtime_error("Output extension must be .png or .svg");
        for (const auto& panel : panels) {
            if (panel.contains("file") && resolve(root, panel.at("file")) == output)
                throw std::runtime_error("Output would overwrite input");
        }
        fs::create_directories(output.parent_path());
        const auto staging = fs::path(output.wstring() + L".partial");
        const auto script = fs::path(output.wstring() + L".gp");
        write(script, "set encoding utf8\nset terminal " + std::string(output.extension() == ".png" ? "pngcairo" : "svg") + settings + "set output " + quote(staging.generic_u8string()) + "\n" + body.str() + "unset output\n");
        launch(exe, script, true);
        if (!fs::exists(staging) || fs::file_size(staging) < 100)
            throw std::runtime_error("Export missing/empty");
        if (fs::exists(output))
            fs::remove(output);
        fs::rename(staging, output);
        std::cout << "Saved: " << output.u8string() << '\n';
    }
    if (forceShow || config.value("show", outputName.empty()) || selected) {
        const auto work = fs::temp_directory_path() / "fdk-quick-plot";
        fs::create_directories(work);
        const auto script = work / ("view-" + std::to_string(GetCurrentProcessId()) + ".gp");
        write(script, "set encoding utf8\nset terminal qt" + settings + body.str() + "set mouse\npause mouse close\n");
        launch(exe, script, false);
        std::cout << "Plot window closed.\n";
    }
}
void examples(const fs::path& dir)
{
    if (fs::exists(dir / "gallery.json"))
        throw std::runtime_error("Examples already exist; choose a new folder");
    fs::create_directories(dir);
    std::ostringstream curves, matrix;
    curves << "x,sin,cos,z\n";
    for (int i = 0; i <= 200; ++i) {
        const double x = i * .05;
        curves << x << ',' << std::sin(x) << ',' << std::cos(x) << ',' << x * .1 << '\n';
    }
    for (int y = 0; y < 41; ++y) {
        for (int x = 0; x < 41; ++x) {
            if (x)
                matrix << ',';
            const double a = (x - 20) * .15, b = (y - 20) * .15;
            matrix << std::exp(-.2 * (a * a + b * b)) * std::cos(a) * std::sin(b);
        }
        matrix << '\n';
    }
    write(dir / "curves.csv", curves.str());
    write(dir / "matrix.csv", matrix.str());
    J line = { { "type", "line" }, { "title", "Curve overlay" }, { "xlabel", "x" }, { "ylabel", "value" }, { "series", J::array({ J { { "file", "curves.csv" }, { "x", "x" }, { "y", "sin" }, { "name", "sin(x)" } }, J { { "file", "curves.csv" }, { "x", "x" }, { "y", "cos" }, { "name", "cos(x)" }, { "dash", 2 } } }) } };
    J grid = { { "file", "matrix.csv" }, { "x0", -3 }, { "y0", -3 }, { "dx", .15 }, { "dy", .15 } };
    J heat = grid, surface = grid, contour = grid;
    heat["type"] = "heatmap";
    surface["type"] = "surface";
    contour["type"] = "contour";
    J scatter = { { "type", "scatter3d" }, { "title", "3D points" }, { "file", "curves.csv" }, { "x", "sin" }, { "y", "cos" }, { "z", "z" } };
    J panels = J::array({ line, heat, surface, scatter, contour });
    J points = line;
    points["type"] = "scatter";
    points["title"] = "2D points";
    panels.push_back(points);
    J config = { { "rows", 2 }, { "cols", 3 }, { "width", 1500 }, { "height", 900 }, { "title", "Quick plot examples" }, { "output", "gallery.png" }, { "show", false }, { "panels", panels } };
    write(dir / "gallery.json", config.dump(2));
    std::cout << "Created: " << (dir / "gallery.json").u8string() << '\n';
}
int programMain(int argc, char** argv)
{
    try {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "quick_plot config.json [--show] [--panel N]\nquick_plot --examples directory\nquick_plot --line data.csv --x x --y y1,y2 --output figure.png [--show]\nquick_plot --heatmap matrix.csv --output image.png\nquick_plot --surface matrix.csv --show\nTypes in JSON: line, scatter, heatmap, contour, surface, scatter3d.\n";
            return 0;
        }
        if (std::string(argv[1]) == "--examples") {
            if (argc != 3)
                throw std::runtime_error("--examples requires a directory");
            examples(fs::u8path(argv[2]));
            return 0;
        }
        bool show = false;
        int selected = 0;
        J config;
        fs::path root;
        const std::string first = argv[1];
        if (first.rfind("--", 0) == 0) {
            if (argc < 3)
                throw std::runtime_error("Missing CSV path");
            J panel = { { "type", first.substr(2) }, { "file", fs::absolute(fs::u8path(argv[2])).generic_u8string() } };
            J ys;
            for (int i = 3; i < argc; ++i) {
                const std::string key = argv[i];
                if (key == "--show") {
                    show = true;
                    continue;
                }
                if (i + 1 >= argc)
                    throw std::runtime_error("Missing option value");
                const std::string value = argv[++i];
                if (key == "--output")
                    config["output"] = value;
                else if (key == "--title")
                    panel["title"] = value;
                else if (key == "--x" || key == "--z")
                    panel[key.substr(2)] = value;
                else if (key == "--y")
                    ys = split(value);
                else
                    throw std::runtime_error("Unknown option: " + key);
            }
            if (!ys.is_null()) {
                J series = J::array();
                for (const auto& y : ys) {
                    auto s = panel;
                    s["y"] = y;
                    s["name"] = y;
                    series.push_back(s);
                }
                panel["series"] = series;
            }
            config["panels"] = J::array({ panel });
            root = fs::current_path();
        } else {
            const auto file = fs::absolute(fs::u8path(first));
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("Cannot read config");
            in >> config;
            root = file.parent_path();
            for (int i = 2; i < argc; ++i) {
                const std::string key = argv[i];
                if (key == "--show")
                    show = true;
                else if (key == "--panel" && i + 1 < argc) {
                    selected = std::stoi(argv[++i]);
                    if (selected < 1)
                        throw std::runtime_error("--panel starts at 1");
                } else
                    throw std::runtime_error("Unknown option: " + key);
            }
        }
        render(config, root, show, selected);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "quick_plot: " << e.what() << '\n';
        return 1;
    }
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
    std::vector<std::string> utf8;
    for (int i = 0; i < argc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(size_t(n), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
        s.pop_back();
        utf8.push_back(std::move(s));
    }
    std::vector<char*> args;
    for (auto& s : utf8)
        args.push_back(s.data());
    return programMain(argc, args.data());
}
#else
int main(int argc, char** argv) { return programMain(argc, argv); }
#endif

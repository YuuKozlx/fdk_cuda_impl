#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <matplot/matplot.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Channel = std::vector<std::vector<unsigned char>>;
Json readJson(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open " + path.string());
    return Json::parse(in);
}
template <class T>
std::vector<T> readSlice(const fs::path& path, const std::array<int, 3>& dims, int slice = -1)
{
    const size_t plane = size_t(dims[0]) * dims[1];
    if (fs::file_size(path) != plane * dims[2] * sizeof(T))
        throw std::runtime_error("RAW dimensions mismatch: " + path.string());
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    if (slice < 0)
        slice = dims[2] / 2;
    if (slice >= dims[2])
        throw std::runtime_error("Invalid slice index");
    in.seekg(plane * slice * sizeof(T));
    std::vector<T> data(plane);
    in.read(reinterpret_cast<char*>(data.data()), plane * sizeof(T));
    for (auto value : data)
        if (!std::isfinite(double(value)))
            throw std::runtime_error("Nonfinite image: " + path.string());
    return data;
}
template<class T=float>
std::vector<T> readPlane(const fs::path& path,const std::array<int,3>& dims,int index,int plane) {
    if(plane==0) return readSlice<T>(path,dims,index);
    if(plane<1 || plane>2 || index<0 || index>=dims[plane==1?1:0])
        throw std::runtime_error("Invalid orthogonal plane");
    const size_t slab=size_t(dims[0])*dims[1];
    if(fs::file_size(path)!=slab*dims[2]*sizeof(T)) throw std::runtime_error("Invalid volume size");
    const int width=plane==1?dims[0]:dims[1];
    std::vector<T> values(size_t(width)*dims[2]), row(plane==1?dims[0]:slab);
    std::ifstream in(path,std::ios::binary); in.exceptions(std::ios::failbit|std::ios::badbit);
    for(int z=0;z<dims[2];++z) {
        in.seekg(std::streamoff((size_t(z)*slab+(plane==1?size_t(index)*dims[0]:0))*sizeof(T)));
        in.read(reinterpret_cast<char*>(row.data()),row.size()*sizeof(T));
        for(int x=0;x<width;++x) {
            const T value=plane==1?row[x]:row[size_t(x)*dims[0]+index];
            if(!std::isfinite(value)) throw std::runtime_error("Nonfinite orthogonal image");
            values[size_t(dims[2]-1-z)*width+x]=value;
        }
    }
    return values;
}
template <class F>
void draw(const matplot::axes_handle& ax, int nx, int ny, F pixel, const std::string& title)
{
    Channel r(ny, std::vector<unsigned char>(nx)), g = r, b = r;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const auto c = pixel(size_t(y) * nx + x);
            r[y][x] = static_cast<unsigned char>(std::lround(std::clamp(c[0], 0., 1.) * 255));
            g[y][x] = static_cast<unsigned char>(std::lround(std::clamp(c[1], 0., 1.) * 255));
            b[y][x] = static_cast<unsigned char>(std::lround(std::clamp(c[2], 0., 1.) * 255));
        }
    ax->imshow(r, g, b);
    ax->title(title);
    ax->font_size(13);
}
void save(const fs::path& file)
{
    const fs::path staging = file.string() + "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".partial.png";
    const bool sent = matplot::save(staging.generic_string(), "pngcairo");
    // Matplot++ 1.2.1's Windows hiddenPclose waits on a closed pipe handle,
    // not the child process. Validate PNG completion rather than assuming save
    // or backend destruction waits for the final file write.
    matplot::gcf()->backend(nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    const std::array<unsigned char, 12> pngEnd { 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };
    bool complete = false;
    while (sent && std::chrono::steady_clock::now() < deadline) {
        std::ifstream in(staging, std::ios::binary | std::ios::ate);
        if (in && in.tellg() >= 20) {
            std::array<unsigned char, 12> end {};
            in.seekg(-12, std::ios::end);
            in.read(reinterpret_cast<char*>(end.data()), end.size());
            if (in && end == pngEnd) {
                complete = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!complete)
        throw std::runtime_error("Matplot++/Gnuplot export failed: " + file.string());
    if (fs::exists(file))
        fs::remove(file);
    fs::rename(staging, file);
}
// Native voxel images need only window mapping, not a Gnuplot mesh. Export
// the diagnostic comparison directly, retaining every input pixel.
void boneWindowComparison(const fs::path& dir, const std::array<int, 3>& dims,
    double mu, const std::vector<int>& slices, double width, double level,
    const std::string& filename, std::array<int, 4> crop = {}, int scale = 1, int plane = 0,
    bool bilateral = false, bool sampling = false, const Json* comparison = nullptr)
{
#ifdef _WIN32
    const std::array<std::string, 3> names = comparison ?
        std::array<std::string, 3>{ comparison->at("columns").at(0).at("directory").get<std::string>(),
            comparison->at("columns").at(1).at("directory").get<std::string>(), comparison->at("columns").at(2).at("directory").get<std::string>() } : sampling ?
        std::array<std::string, 3>{ "mono-reference", "mono-reference-spp16", "source-materials" } : bilateral ?
        std::array<std::string, 3>{ "bone-bhc", "bone-bhc-bilateral2d", "mono-reference" } :
        std::array<std::string, 3>{ "water-bhc", "bone-bhc", "mono-reference" };
    const std::array<std::string, 3> captions = comparison ?
        std::array<std::string, 3>{ comparison->at("columns").at(0).at("title").get<std::string>(),
            comparison->at("columns").at(1).at("title").get<std::string>(), comparison->at("columns").at(2).at("title").get<std::string>() } : sampling ?
        std::array<std::string, 3>{ "Mono / 1 random ray", "Mono / 16 random rays", "Source materials / no reconstruction" } : bilateral ?
        std::array<std::string, 3>{ "Bone BHC / unfiltered", "Bone BHC + bilateral 2D", "Mono / unfiltered" } :
        std::array<std::string, 3>{ "Scatter + water BHC", "Scatter + water + bone BHC", "Mono / same reconstruction" };
    const int sourceX=plane==2?dims[1]:dims[0],sourceY=plane==0?dims[1]:dims[2];
    if (crop[2] == 0 && crop[3] == 0) crop = { 0, 0, sourceX, sourceY };
    if (scale <= 0 || crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 || crop[0] + crop[2] > sourceX || crop[1] + crop[3] > sourceY)
        throw std::runtime_error("Invalid bone-window detail crop");
    const int imageX = crop[2] * scale, imageY = crop[3] * scale;
    const int header = 48, gap = 8;
    const int nx = 3 * imageX + 2 * gap;
    const int ny = int(slices.size()) * (imageY + header + gap) - gap;
    BITMAPINFO info {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = nx;
    info.bmiHeader.biHeight = -ny;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* memory = nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &memory, nullptr, 0);
    if (!dc || !bitmap || !memory) {
        if (bitmap) DeleteObject(bitmap);
        if (dc) DeleteDC(dc);
        throw std::runtime_error("Cannot allocate bone-window bitmap");
    }
    const auto previousBitmap = SelectObject(dc, bitmap);
    HFONT font = CreateFontA(-18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
        DEFAULT_PITCH, "Segoe UI");
    const auto previousFont = SelectObject(dc, font);
    auto* pixels = static_cast<unsigned char*>(memory);
    const size_t bytes = size_t(nx) * ny * 4;
    std::fill(pixels, pixels + bytes, 255);
    SetTextColor(dc, RGB(0, 0, 0));
    SetBkMode(dc, TRANSPARENT);
    try {
        for (int row = 0; row < int(slices.size()); ++row) {
            const int beginY = row * (imageY + header + gap);
            for (int col = 0; col < 3; ++col) {
                std::vector<float> values;
                if(sampling && col==2) {
                    const auto metrics=readJson(dir/"metrics.json");
                    const auto metadata=readJson(dir/"segmentation/segmentation.json");
                    const auto labels=readPlane<uint8_t>(metrics.at("source_labels").get<std::string>(),dims,slices[row],plane);
                    values.reserve(labels.size());
                    for(auto label:labels) values.push_back(metadata.at("original_material_mu_mm_inv").at(std::to_string(label)).get<float>());
                } else {
                    values=readPlane(dir / names[col] / "reconstruction.raw", dims, slices[row], plane);
                }
                const int beginX = col * (imageX + gap);
                const auto caption = captions[col] + (plane==0?" / z=":plane==1?" / coronal y=":" / sagittal x=") + std::to_string(slices[row]);
                const auto window = "WW " + std::to_string(int(width)) + " HU / WL " + std::to_string(int(level)) + " HU";
                TextOutA(dc, beginX + 8, beginY + 2, caption.c_str(), int(caption.size()));
                TextOutA(dc, beginX + 8, beginY + 24, window.c_str(), int(window.size()));
                for (int y = 0; y < imageY; ++y)
                    for (int x = 0; x < imageX; ++x) {
                        const double hu = values[size_t(crop[1] + y / scale) * sourceX + crop[0] + x / scale] * 1000. / mu - 1000.;
                        const auto gray = static_cast<unsigned char>(std::lround(255 * std::clamp((hu - level + width / 2) / width, 0., 1.)));
                        const size_t index = (size_t(beginY + header + y) * nx + beginX + x) * 4;
                        pixels[index] = pixels[index + 1] = pixels[index + 2] = gray;
                    }
            }
        }
        GdiFlush();
        BITMAPFILEHEADER fileHeader {};
        fileHeader.bfType = 0x4d42;
        fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
        fileHeader.bfSize = DWORD(fileHeader.bfOffBits + bytes);
        std::ofstream out(dir / filename, std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
        out.write(reinterpret_cast<const char*>(&info.bmiHeader), sizeof(info.bmiHeader));
        out.write(reinterpret_cast<const char*>(pixels), bytes);
        out.close();
    } catch (...) {
        SelectObject(dc, previousFont);
        SelectObject(dc, previousBitmap);
        DeleteObject(font);
        DeleteObject(bitmap);
        DeleteDC(dc);
        throw;
    }
    SelectObject(dc, previousFont);
    SelectObject(dc, previousBitmap);
    DeleteObject(font);
    DeleteObject(bitmap);
    DeleteDC(dc);
#else
    throw std::runtime_error("Native bone-window comparison currently requires Windows");
#endif
}
int main(int argc, char** argv)
{
    try {
        if (argc == 4 && std::string(argv[1]) == "--label-overview") {
            const fs::path input(argv[2]);
            const auto bytes = fs::file_size(input);
            if (bytes % (181 * 217) != 0)
                throw std::runtime_error("Expected BrainWeb 181x217xN uint8 labels");
            const std::array<int, 3> dims { 181, 217, int(bytes / (181 * 217)) };
            auto overview = matplot::figure(true);
            overview->size(1800, 1100);
            const std::array<int, 8> slices { 16, 26, 36, 46, 56, 66, 76, 90 };
            for (int k = 0; k < 8; ++k) {
                const auto data = readSlice<uint8_t>(input, dims, slices[k]);
                draw(matplot::subplot(2, 4, k), 181, 217, [&](size_t i) {
                    double v=data[i]==0?0.:data[i]==7?1.:.4;
                    return std::array<double,3>{v,v,v}; }, "Source z=" + std::to_string(slices[k]) + " (0-based)\\nBone white / Soft tissue gray");
            }
            save(fs::absolute(argv[3]));
            return 0;
        }
        const bool tvComparison = argc == 3 && std::string(argv[1]) == "--tv-comparison";
        const bool boneComparison = argc == 3 && std::string(argv[1]) == "--bone-window";
        const bool artifactComparison = argc == 3 && std::string(argv[1]) == "--artifact-window";
        const bool bilateralComparison = argc == 3 && std::string(argv[1]) == "--bilateral-window";
        const bool samplingComparison = argc == 3 && std::string(argv[1]) == "--sampling-window";
        const bool customComparison = argc == 4 && std::string(argv[1]) == "--compare";
        if (argc != 2 && !tvComparison && !boneComparison && !artifactComparison && !bilateralComparison && !samplingComparison && !customComparison)
            throw std::runtime_error("Usage: multispectrum_plot_bone_bhc experiment-directory");
        const fs::path dir = fs::absolute(argv[tvComparison || boneComparison || artifactComparison || bilateralComparison || samplingComparison || customComparison ? 2 : 1]);
        const auto metadata = readJson(dir / "segmentation/segmentation.json");
        const auto metrics = readJson(dir / "metrics.json");
        const auto dims = metadata.at("dimensions_xyz").get<std::array<int, 3>>();
        for (int n : dims)
            if (n <= 0)
                throw std::runtime_error("Invalid dimensions");
        const double mu = metadata.at("mu_water_mm_inv").get<double>();
        if (!std::isfinite(mu) || mu <= 0)
            throw std::runtime_error("Invalid water reference");
        if(customComparison) {
            const auto comparison=readJson(fs::absolute(argv[3]));
            if(comparison.at("columns").size()!=3) throw std::runtime_error("Expected three comparison columns");
            const auto file=comparison.at("filename").get<std::string>();
            if(fs::path(file).filename()!=fs::path(file) || fs::path(file).extension()!=".bmp")
                throw std::runtime_error("Comparison output must be a BMP filename inside the experiment directory");
            const double width=comparison.at("window_width_hu").get<double>(), level=comparison.at("window_level_hu").get<double>();
            if(!std::isfinite(width) || width<=0 || !std::isfinite(level)) throw std::runtime_error("Invalid display window");
            boneWindowComparison(dir,dims,mu,comparison.at("indices").get<std::vector<int>>(),width,level,file,
                comparison.value("crop_xywh",std::array<int,4>{}),comparison.value("scale",1),comparison.value("plane",0),false,false,&comparison);
            return 0;
        }
        if(samplingComparison) {
            const int base=metrics.value("display_z",dims[2]/2);
            boneWindowComparison(dir,dims,mu,{std::min(base+24,dims[2]-1)},250,25,"sampling-skullbase-ww250-wl25.bmp",{},1,0,false,true);
            boneWindowComparison(dir,dims,mu,{base},250,25,"sampling-skullbase-lower-ww250-wl25.bmp",{},1,0,false,true);
            boneWindowComparison(dir,dims,mu,{dims[1]/2},250,25,"sampling-coronal-ww250-wl25.bmp",{},1,1,false,true);
            boneWindowComparison(dir,dims,mu,{dims[0]/2},250,25,"sampling-sagittal-ww250-wl25.bmp",{},1,2,false,true);
            std::cout << "Sampling diagnostic comparisons exported\n";
            return 0;
        }
        if(bilateralComparison) {
            const int base=metrics.value("display_z",dims[2]/2);
            boneWindowComparison(dir,dims,mu,{std::min(base+24,dims[2]-1)},250,25,"bilateral-skullbase-ww250-wl25.bmp",{},1,0,true);
            boneWindowComparison(dir,dims,mu,{dims[1]/2},250,25,"bilateral-coronal-ww250-wl25.bmp",{},1,1,true);
            boneWindowComparison(dir,dims,mu,{dims[0]/2},250,25,"bilateral-sagittal-ww250-wl25.bmp",{},1,2,true);
            if(dims[0]>=244 && dims[1]>=368)
                boneWindowComparison(dir,dims,mu,{base,std::min(base+24,dims[2]-1)},1000,700,
                    "bilateral-bone-detail-ww1000-wl700.bmp",{100,224,144,144},2,0,true);
            std::cout << "Bilateral window comparisons exported\n";
            return 0;
        }
        if(artifactComparison) {
            const int base=metrics.value("display_z",dims[2]/2);
            boneWindowComparison(dir,dims,mu,{base,std::min(base+24,dims[2]-1),std::min(base+48,dims[2]-1)},250,25,"artifact-axial-ww250-wl25.bmp");
            boneWindowComparison(dir,dims,mu,{std::min(base+24,dims[2]-1)},250,25,"artifact-skullbase-ww250-wl25.bmp");
            boneWindowComparison(dir,dims,mu,{dims[1]/2},250,25,"artifact-coronal-ww250-wl25.bmp",{},1,1);
            boneWindowComparison(dir,dims,mu,{dims[0]/2},250,25,"artifact-sagittal-ww250-wl25.bmp",{},1,2);
            std::cout << "Native orthogonal artifact comparisons exported\n";
            return 0;
        }
        if (boneComparison) {
            const int base = metrics.value("display_z", dims[2] / 2);
            const std::vector<int> slices { base, std::min(base + 24, dims[2] - 1), std::min(base + 48, dims[2] - 1) };
            boneWindowComparison(dir, dims, mu, slices, 1500, 500, "bone-window-ww1500-wl500.bmp");
            boneWindowComparison(dir, dims, mu, slices, 1000, 700, "bone-window-ww1000-wl700.bmp");
            if (dims[0] >= 244 && dims[1] >= 368)
                boneWindowComparison(dir, dims, mu, { slices[0], slices[1] }, 1000, 700,
                    "bone-detail-ww1000-wl700.bmp", { 100, 224, 144, 144 }, 2);
            std::ofstream manifest(dir / "bone-window.json");
            manifest << Json { { "slices_z", slices }, { "dimensions_xyz", dims },
                { "mu_water_mm_inv", mu }, { "columns", { "water-bhc", "bone-bhc", "mono-reference" } },
                { "windows", { { 1500, 500 }, { 1000, 700 } } },
                { "detail_crop_xywh", { 100, 224, 144, 144 } }, { "detail_pixel_replication", 2 },
                { "policy", "Native voxel sampling; same TV; display remapping only" } }.dump(2);
            std::cout << "Bone-window comparisons exported\n";
            return 0;
        }
        if (tvComparison) {
            const std::array<std::string, 3> cases { "bone-bhc", "bone-bhc-tv-1e-5", "mono-reference" };
            const std::array<std::string, 4> headings { "Poly + bone / TV 1e-4", "Poly + bone / TV 1e-5", "Mono / TV 1e-4", "Source materials / no reconstruction" };
            const std::array<int, 1> slices { dims[2] / 2 };
            auto figure = matplot::figure(true);
            figure->size(1900, 600);
            // Display a central detail crop at native voxel sampling. This does
            // not resample or modify any reconstructed volume.
            const int cropX = std::min(256, dims[0]), cropY = std::min(256, dims[1]);
            const int beginX = (dims[0] - cropX) / 2, beginY = (dims[1] - cropY) / 2;
            for (int row = 0; row < int(slices.size()); ++row) {
                const auto labels = readSlice<uint8_t>(metrics.at("source_labels").get<std::string>(), dims, slices[row]);
                for (int col = 0; col < 4; ++col) {
                    std::vector<float> values;
                    if (col < 3)
                        values = readSlice<float>(dir / cases[col] / "reconstruction.raw", dims, slices[row]);
                    draw(matplot::subplot(1, 4, col), cropX, cropY, [&](size_t i) {
                        const size_t index = (i / cropX + beginY) * dims[0] + i % cropX + beginX;
                        const double attenuation = col < 3 ? values[index] : metadata.at("original_material_mu_mm_inv").at(std::to_string(labels[index])).get<double>();
                        const double gray = (attenuation * 1000 / mu - 1000 + 5) / 80;
                        return std::array<double, 3> { gray, gray, gray };
                    }, headings[col]);
                }
            }
            save(dir / "tv-strength-matplot.png");
            return 0;
        }
        const bool full = metrics.value("simulation_profile", std::string {}) == "full-physics";
        std::vector<std::string> names = { "uncorrected", "water-bhc", "bone-bhc", "mono-reference" };
        std::vector<std::string> titles = { "Poly: uncorrected", "Poly: water BHC", "Poly: water + bone BHC", "Mono 70 keV: reference only" };
        if (full) {
            names.insert(names.begin() + 1, "water-only");
            titles = { "Poly + physics: uncorrected", "Poly: water BHC only", "Poly: scatter + water BHC", "Poly: scatter + water + bone", "Mono 70 keV: reference only" };
        }
        if (metrics.contains("comparison_cases")) {
            names.clear();
            titles.clear();
            for (const auto& item : metrics.at("comparison_cases")) {
                names.push_back(item.at("name").get<std::string>());
                titles.push_back(item.at("title").get<std::string>());
            }
            if (names.empty())
                throw std::runtime_error("Empty comparison cases");
        }
        const int displayZ = metrics.value("display_z", dims[2] / 2);
        const auto reference = readSlice<float>(dir / "mono-reference/reconstruction.raw", dims, displayZ);
        const int cols = int(names.size());
        auto fig = matplot::figure(true);
        fig->size(cols * 420, 1040);
        for (int c = 0; c < cols; ++c) {
            const auto data = readSlice<float>(dir / names[c] / "reconstruction.raw", dims, displayZ);
            draw(matplot::subplot(2, cols, c), dims[0], dims[1], [&](size_t i) {
                double gray=(data[i]*1000/mu-1000+100)/250;
                return std::array<double,3>{gray,gray,gray}; }, titles[c] + "\\nWL 25 / WW 250 HU");
            draw(matplot::subplot(2, cols, cols + c), dims[0], dims[1], [&](size_t i) {
                double t=std::clamp((data[i]-reference[i])*1000/mu/150.,-1.,1.);
                return t>=0 ? std::array<double,3>{1.,1.-t,1.-t} : std::array<double,3>{1.+t,1.+t,1.}; }, "Error vs mono (HU)\\nBlue -150 / White 0 / Red +150");
        }
        save(dir / "comparison-matplot.png");
        auto segmentation = readSlice<uint8_t>(dir / "segmentation/segmented-u8.raw", dims, displayZ);
        auto truth = readSlice<uint8_t>(metrics.at("source_labels").get<std::string>(), dims, displayZ);
        fig = matplot::figure(true);
        fig->size(1000, 650);
        draw(matplot::subplot(1, 2, 0), dims[0], dims[1], [&](size_t i) {
            if(segmentation[i]==0) return std::array<double,3>{0,0,0};
            if(segmentation[i]==1) return std::array<double,3>{.3,.65,.8};
            return std::array<double,3>{1,.85,.2}; }, "Reconstruction-derived labels\\nBlack: air / Blue: soft / Yellow: bone");
        draw(matplot::subplot(1, 2, 1), dims[0], dims[1], [&](size_t i) {
            const double v=truth[i]==7?1.:0.;return std::array<double,3>{v,v,v}; }, "Original skull labels\\nEvaluation only");
        save(dir / "segmentation-matplot.png");
        if (metrics.at("results").at("bone-bhc").contains("artifact_by_slice")) {
            fig = matplot::figure(true);
            fig->size(1500, 900);
            const std::array<std::string, 4> fields { "lowfreq_rmse_HU", "lowfreq_std_HU", "highfreq_std_HU", "abs_error_gt_40_HU_fraction" };
            const std::array<std::string, 4> labels { "Low-frequency error RMS (HU)", "Low-frequency error variation (HU)", "High-frequency error variation (HU)", "Fraction |error| > 40 HU" };
            for (int p = 0; p < 4; ++p) {
                auto ax = matplot::subplot(2, 2, p);
                for (const auto& name : std::array<std::string, 2> { "water-bhc", "bone-bhc" }) {
                    std::vector<double> x, y;
                    for (const auto& entry : metrics.at("results").at(name).at("artifact_by_slice"))
                        if (entry.contains(fields[p])) {
                            x.push_back(entry.at("z").get<double>());
                            y.push_back(entry.at(fields[p]).get<double>());
                        }
                    ax->plot(x, y)->line_width(1.8);
                    ax->hold(matplot::on);
                }
                ax->title(labels[p]);
                ax->xlabel("Reconstruction slice z (full-volume evaluation)");
                ax->grid(matplot::on);
                ax->legend({ "Scatter + water BHC", "Plus bone BHC" });
            }
            save(dir / "artifact-by-slice-matplot.png");
        }
        if (dims[2] > 33) {
            const size_t n = size_t(dims[0]) * dims[1] * dims[2];
            std::array<std::vector<float>, 3> volumes;
            int k = 0;
            for (const auto& name : std::array<std::string, 3> { "water-bhc", "bone-bhc", "mono-reference" }) {
                const auto path = dir / name / "reconstruction.raw";
                if (fs::file_size(path) != n * sizeof(float))
                    throw std::runtime_error("Invalid volume size");
                std::ifstream in(path, std::ios::binary);
                in.exceptions(std::ios::failbit | std::ios::badbit);
                volumes[k].resize(n);
                in.read(reinterpret_cast<char*>(volumes[k++].data()), n * sizeof(float));
            }
            fig = matplot::figure(true);
            fig->size(1800, 1500);
            for (int plane = 0; plane < 3; ++plane) {
                int nx = plane == 2 ? dims[1] : dims[0], ny = plane == 0 ? dims[1] : dims[2];
                const std::string planeName = plane == 0 ? "Axial z=" + std::to_string(displayZ) : plane == 1 ? "Coronal mid-Y"
                                                                                                              : "Sagittal mid-X";
                auto index = [&](size_t i) {int x=int(i%nx),y=int(i/nx);return plane==0?(size_t(displayZ)*dims[1]+y)*dims[0]+x:
                    plane==1?(size_t(dims[2]-1-y)*dims[1]+dims[1]/2)*dims[0]+x:(size_t(dims[2]-1-y)*dims[1]+x)*dims[0]+dims[0]/2; };
                for (int col = 0; col < 4; ++col) {
                    draw(matplot::subplot(3, 4, plane * 4 + col), nx, ny, [&](size_t i) {
                        const auto j=index(i);
                        if(col<2) {double v=(volumes[col][j]*1000/mu-1000+100)/250;return std::array<double,3>{v,v,v};}
                        double t=std::clamp((volumes[col-2][j]-volumes[2][j])*1000/mu/100.,-1.,1.);
                        return t>=0?std::array<double,3>{1.,1.-t,1.-t}:std::array<double,3>{1.+t,1.+t,1.}; }, planeName + "\\n" + (col == 0 ? "Scatter + water (WW 250 HU)" : col == 1 ? "Plus bone (WW 250 HU)"
                                                                                                     : col == 2                                                       ? "Water error (-100..100 HU)"
                                                                                                                                                                      : "Bone error (-100..100 HU)"));
                }
            }
            save(dir / "whole-volume-orthogonal-matplot.png");
        }
        std::cout << "Matplot++ comparison and segmentation saved in " << dir << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

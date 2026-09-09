#include "XcomAttenuationProvider.hpp"

#include <cmath>
#include <iostream>

int main()
{
    yk::spectral::MaterialSpec water;
    water.label = 1;
    water.name = "water";
    water.formula = "H2O";
    water.density_g_cm3 = 1.0;

    yk::spectral::XcomAttenuationProvider provider(XCOM_TEST_DATA_DIRECTORY);
    std::vector<double> values;
    std::string error;
    if (!provider.query(water, {20.0, 60.0, 100.0}, values, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const double expected[] = {0.809820711613, 0.205873382278, 0.170726725133};
    if (values.size() != 3) return 2;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 2.0e-6) {
            std::cerr << "XCOM water mismatch at " << i << ": " << values[i] << '\n';
            return 3;
        }
    }
    return 0;
}

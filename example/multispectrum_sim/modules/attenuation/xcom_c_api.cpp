#include "xcom_c_api.h"
#include "xcom.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
void ResetResult(XcomResult* result)
{
    if (result != nullptr)
    {
        *result = XcomResult{};
    }
}

void SetError(char* destination, size_t capacity, const char* message)
{
    if (destination != nullptr && capacity > 0)
    {
        std::snprintf(destination, capacity, "%s", message != nullptr ? message : "Unknown XCOM error");
    }
}

void AllocateResult(size_t length, XcomResult* result)
{
    result->length = length;
    result->photon_energy_keV = new double[length];
    result->coherent_scattering = new double[length];
    result->incoherent_scattering = new double[length];
    result->photoelectric_absorption = new double[length];
    result->nuclear_field_pair_production = new double[length];
    result->electron_field_pair_production = new double[length];
    result->total_with_coherent = new double[length];
    result->total_without_coherent = new double[length];
}

int Calculate(
    const char* dataDirectory,
    const std::vector<int>& atomicNumbers,
    const std::vector<double>& massFractions,
    const double* photonEnergyKeV,
    size_t energyCount,
    XcomResult* result,
    char* errorMessage,
    size_t errorMessageCapacity)
{
    ResetResult(result);
    try
    {
        if (result == nullptr)
        {
            throw std::invalid_argument("A result is required");
        }
        if (atomicNumbers.empty() || atomicNumbers.size() != massFractions.size())
        {
            throw std::invalid_argument("Element and mass-fraction arrays must have the same non-zero length");
        }
        if (photonEnergyKeV == nullptr || energyCount == 0)
        {
            throw std::invalid_argument("At least one photon energy is required");
        }

        std::vector<int> z = atomicNumbers;
        std::vector<float> weights(massFractions.size());
        double weightSum = 0.0;
        for (size_t i = 0; i < massFractions.size(); ++i)
        {
            if (z[i] < 1 || z[i] > 100 || !std::isfinite(massFractions[i]) || massFractions[i] < 0.0)
            {
                throw std::invalid_argument("Atomic numbers must be 1..100 and mass fractions must be finite and non-negative");
            }
            weightSum += massFractions[i];
        }
        if (!(weightSum > 0.0))
        {
            throw std::invalid_argument("Mass fractions must have a positive sum");
        }
        for (size_t i = 0; i < massFractions.size(); ++i)
        {
            weights[i] = static_cast<float>(massFractions[i] / weightSum);
        }

        std::vector<float> energyEv(energyCount);
        std::vector<int> edgeFlags(energyCount, -1);
        std::vector<int> energyIndices(energyCount, 0);
        for (size_t i = 0; i < energyCount; ++i)
        {
            if (!std::isfinite(photonEnergyKeV[i]) || photonEnergyKeV[i] <= 0.0)
            {
                throw std::invalid_argument("Photon energies must be positive and finite");
            }
            energyEv[i] = static_cast<float>(photonEnergyKeV[i] * 1000.0);
        }

        std::vector<float> coherent(energyCount);
        std::vector<float> incoherent(energyCount);
        std::vector<float> photoelectric(energyCount);
        std::vector<float> pairNuclear(energyCount);
        std::vector<float> pairElectron(energyCount);
        std::vector<float> edgeDifference(energyCount);

        SetXcomErrorHandler(nullptr);
        SetXcomDataDirectory(dataDirectory != nullptr ? dataDirectory : "");
        Calculation(
            static_cast<int>(z.size()), z.data(), weights.data(), 3, 3,
            static_cast<int>(energyCount), energyEv.data(), edgeFlags.data(), energyIndices.data(),
            coherent.data(), incoherent.data(), photoelectric.data(), pairNuclear.data(),
            pairElectron.data(), edgeDifference.data());

        AllocateResult(energyCount, result);
        for (size_t i = 0; i < energyCount; ++i)
        {
            const double photoelectricBelowEdge = photoelectric[i] - edgeDifference[i];
            result->photon_energy_keV[i] = photonEnergyKeV[i];
            result->coherent_scattering[i] = coherent[i];
            result->incoherent_scattering[i] = incoherent[i];
            result->photoelectric_absorption[i] = photoelectricBelowEdge;
            result->nuclear_field_pair_production[i] = pairNuclear[i];
            result->electron_field_pair_production[i] = pairElectron[i];
            result->total_with_coherent[i] = coherent[i] + incoherent[i] + photoelectricBelowEdge + pairNuclear[i] + pairElectron[i];
            result->total_without_coherent[i] = incoherent[i] + photoelectricBelowEdge + pairNuclear[i] + pairElectron[i];
        }
        SetError(errorMessage, errorMessageCapacity, "");
        return 0;
    }
    catch (const std::exception& error)
    {
        xcom_free_result(result);
        SetError(errorMessage, errorMessageCapacity, error.what());
        return -1;
    }
    catch (...)
    {
        xcom_free_result(result);
        SetError(errorMessage, errorMessageCapacity, "Unknown XCOM error");
        return -1;
    }
}
}

int xcom_calculate_formula(
    const char* dataDirectory,
    const char* formula,
    const double* photonEnergyKeV,
    size_t energyCount,
    XcomResult* result,
    char* errorMessage,
    size_t errorMessageCapacity)
{
    try
    {
        if (formula == nullptr || formula[0] == '\0')
        {
            throw std::invalid_argument("A non-empty chemical formula is required");
        }
        std::vector<int> atomicNumbers;
        std::vector<double> atoms;
        std::vector<double> massFractions;
        if (!ParseChemicalFormula(formula, atomicNumbers, atoms))
        {
            throw std::invalid_argument("Invalid chemical formula");
        }
        for (int atomicNumber : atomicNumbers)
        {
            if (atomicNumber < 1 || atomicNumber > 100)
            {
                throw std::invalid_argument("XCOM data supports atomic numbers 1 through 100");
            }
        }
        if (!GetFractionByWeight(atomicNumbers, atoms, massFractions))
        {
            throw std::invalid_argument("Invalid chemical formula");
        }
        return Calculate(dataDirectory, atomicNumbers, massFractions, photonEnergyKeV, energyCount, result, errorMessage, errorMessageCapacity);
    }
    catch (const std::exception& error)
    {
        ResetResult(result);
        SetError(errorMessage, errorMessageCapacity, error.what());
        return -1;
    }
}

int xcom_calculate_mass_fractions(
    const char* dataDirectory,
    const int* atomicNumbers,
    const double* massFractions,
    size_t elementCount,
    const double* photonEnergyKeV,
    size_t energyCount,
    XcomResult* result,
    char* errorMessage,
    size_t errorMessageCapacity)
{
    if (atomicNumbers == nullptr || massFractions == nullptr || elementCount == 0)
    {
        ResetResult(result);
        SetError(errorMessage, errorMessageCapacity, "Element arrays must be non-null and non-empty");
        return -1;
    }
    return Calculate(
        dataDirectory,
        std::vector<int>(atomicNumbers, atomicNumbers + elementCount),
        std::vector<double>(massFractions, massFractions + elementCount),
        photonEnergyKeV, energyCount, result, errorMessage, errorMessageCapacity);
}

void xcom_free_result(XcomResult* result)
{
    if (result == nullptr)
    {
        return;
    }
    delete[] result->photon_energy_keV;
    delete[] result->coherent_scattering;
    delete[] result->incoherent_scattering;
    delete[] result->photoelectric_absorption;
    delete[] result->nuclear_field_pair_production;
    delete[] result->electron_field_pair_production;
    delete[] result->total_with_coherent;
    delete[] result->total_without_coherent;
    ResetResult(result);
}

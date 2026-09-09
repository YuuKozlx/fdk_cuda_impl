#ifndef XCOM_C_API_H
#define XCOM_C_API_H

#include <stddef.h>

#if defined(XCOM_C_API_STATIC)
#define XCOM_API
#elif defined(_WIN32) && defined(XCOM_C_API_BUILD)
#define XCOM_API __declspec(dllexport)
#elif defined(_WIN32)
#define XCOM_API __declspec(dllimport)
#else
#define XCOM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XcomResult
{
    size_t length;
    double* photon_energy_keV;
    double* coherent_scattering;
    double* incoherent_scattering;
    double* photoelectric_absorption;
    double* nuclear_field_pair_production;
    double* electron_field_pair_production;
    double* total_with_coherent;
    double* total_without_coherent;
} XcomResult;

XCOM_API int xcom_calculate_formula(
    const char* data_directory,
    const char* formula,
    const double* photon_energy_keV,
    size_t energy_count,
    XcomResult* result,
    char* error_message,
    size_t error_message_capacity);

XCOM_API int xcom_calculate_mass_fractions(
    const char* data_directory,
    const int* atomic_numbers,
    const double* mass_fractions,
    size_t element_count,
    const double* photon_energy_keV,
    size_t energy_count,
    XcomResult* result,
    char* error_message,
    size_t error_message_capacity);

XCOM_API void xcom_free_result(XcomResult* result);

#ifdef __cplusplus
}
#endif

#endif

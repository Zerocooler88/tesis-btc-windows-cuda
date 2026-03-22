from __future__ import annotations

from pathlib import Path
import polars as pl


RUTA_GLOBAL = Path("data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet")

CARPETA_REGIMENES = Path("data/1h/regimenes")
CARPETA_TRIMESTRES = Path("data/1h/trimestres")
CARPETA_EXTREMOS = Path("data/1h/extremos")
CARPETA_RESUMENES = Path("data/1h/resumenes")

for carpeta in [
    CARPETA_REGIMENES,
    CARPETA_TRIMESTRES,
    CARPETA_EXTREMOS,
    CARPETA_RESUMENES,
]:
    carpeta.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not RUTA_GLOBAL.exists():
        raise FileNotFoundError(f"No existe el dataset global: {RUTA_GLOBAL}")

    print("=" * 70)
    print("GENERACION DE ESCENARIOS 1H")
    print(f"Leyendo: {RUTA_GLOBAL}")

    df = pl.read_parquet(RUTA_GLOBAL).sort("open_datetime_utc")

    columnas_obligatorias = [
        "open_datetime_utc",
        "year_quarter",
        "regimen_mercado",
        "close",
    ]
    faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

    print(f"Filas leidas: {df.height:,}")
    print(f"Columnas leidas: {df.width}")

    # --------------------------------------------------
    # 1) Regimenes
    # --------------------------------------------------
    for regimen in ["pre_covid", "covid", "post_covid"]:
        df_reg = df.filter(pl.col("regimen_mercado") == regimen).sort(
            "open_datetime_utc"
        )
        ruta_reg = CARPETA_REGIMENES / f"btcusdt_spot_1h_{regimen}.parquet"
        df_reg.write_parquet(ruta_reg, compression="zstd")
        print("-" * 70)
        print(f"Regimen: {regimen}")
        print(f"Filas: {df_reg.height:,}")
        print(f"Archivo: {ruta_reg}")

    # --------------------------------------------------
    # 2) Trimestres individuales
    # --------------------------------------------------
    trimestres = (
        df.select("year_quarter").unique().sort("year_quarter").to_series().to_list()
    )

    for trimestre in trimestres:
        df_trim = df.filter(pl.col("year_quarter") == trimestre).sort(
            "open_datetime_utc"
        )
        ruta_trim = (
            CARPETA_TRIMESTRES / f"btcusdt_spot_1h_{trimestre}_model_input.parquet"
        )
        df_trim.write_parquet(ruta_trim, compression="zstd")
        print("-" * 70)
        print(f"Trimestre: {trimestre}")
        print(f"Filas: {df_trim.height:,}")
        print(f"Archivo: {ruta_trim}")

    # --------------------------------------------------
    # 3) Resumen trimestral para hallar extremos
    # --------------------------------------------------
    resumen_trim = (
        df.group_by("year_quarter")
        .agg(
            [
                pl.len().alias("filas"),
                pl.col("close").mean().alias("close_promedio"),
                pl.col("close").min().alias("close_minimo"),
                pl.col("close").max().alias("close_maximo"),
                pl.col("open_datetime_utc").min().alias("fecha_min"),
                pl.col("open_datetime_utc").max().alias("fecha_max"),
            ]
        )
        .sort("year_quarter")
    )

    ruta_resumen = CARPETA_RESUMENES / "resumen_trimestres_close.csv"
    resumen_trim.write_csv(ruta_resumen)
    print("-" * 70)
    print(f"Resumen trimestral guardado en: {ruta_resumen}")

    # --------------------------------------------------
    # 4) Trimestre mas bajo y trimestre mas alto
    #    criterio: close_promedio
    # --------------------------------------------------
    trimestre_bajo = (
        resumen_trim.sort("close_promedio")
        .select("year_quarter")
        .to_series()
        .to_list()[0]
    )

    trimestre_alto = (
        resumen_trim.sort("close_promedio", descending=True)
        .select("year_quarter")
        .to_series()
        .to_list()[0]
    )

    df_bajo = df.filter(pl.col("year_quarter") == trimestre_bajo).sort(
        "open_datetime_utc"
    )
    df_alto = df.filter(pl.col("year_quarter") == trimestre_alto).sort(
        "open_datetime_utc"
    )

    ruta_bajo = (
        CARPETA_EXTREMOS
        / f"btcusdt_spot_1h_trimestre_mas_bajo_{trimestre_bajo}.parquet"
    )
    ruta_alto = (
        CARPETA_EXTREMOS
        / f"btcusdt_spot_1h_trimestre_mas_alto_{trimestre_alto}.parquet"
    )

    df_bajo.write_parquet(ruta_bajo, compression="zstd")
    df_alto.write_parquet(ruta_alto, compression="zstd")

    print("-" * 70)
    print(f"Trimestre mas bajo por close_promedio: {trimestre_bajo}")
    print(f"Archivo: {ruta_bajo}")
    print(f"Trimestre mas alto por close_promedio: {trimestre_alto}")
    print(f"Archivo: {ruta_alto}")

    print("=" * 70)
    print("ESCENARIOS GENERADOS CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()

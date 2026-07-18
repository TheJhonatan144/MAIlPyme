# Evidencias del modelo BETO v2

## Dataset

- Total: 840 correos
- Train: 672
- Validation: 84
- Test: 84
- División estratificada: 80/10/10
- Categorías oficiales: 6

## Configuración

- Modelo base: BETO cased
- Max length: 128
- Learning rate: 2e-5
- Épocas: 5
- Batch por GPU: 8
- Acumulación de gradientes: 2
- Batch efectivo: 16
- FP16: activado
- Métrica principal: F1 macro
- Pesos por clase: activados

## Evaluación interna

- Accuracy: 1.0000
- F1 macro: 1.0000
- F1 ponderado: 1.0000
- Errores: 0 de 84
- Confianza promedio: 0.98327

## Rendimiento en RTX 4060 Laptop

- Latencia media individual: 7.10 ms
- Latencia P95: 9.00 ms
- Throughput batch 1: 145.15 correos/s
- Throughput batch 16: 1396.83 correos/s
- Estabilidad: 100 % en 500 ejecuciones
- Excepciones: 0

## Limitación

El resultado de 100 % se obtuvo sobre un conjunto de prueba interno de 84 correos. Debe interpretarse como evidencia del ajuste al dataset construido, no como garantía de desempeño perfecto en producción.

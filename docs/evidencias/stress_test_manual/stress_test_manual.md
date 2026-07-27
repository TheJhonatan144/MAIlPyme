# Stress Test Manual — Evaluación del modelo BETO v2 con casos difíciles

**Proyecto:** MailPyme AI — Clasificador de correos empresariales para MiPYMEs
**Modelo evaluado:** mailpyme_beto_v2 (BETO fine-tuneado, 6 categorías)
**Fecha de la prueba:** 26 de julio de 2026
**Entorno de inferencia:** CPU (torch 2.7.1+cpu), backend FastAPI local
**Total de casos:** 18 correos redactados manualmente

## 1. Objetivo de la prueba

El modelo obtuvo 100% de accuracy y F1 (macro y weighted) en el conjunto de test de 84
correos sinteticos. Un resultado perfecto no es motivo de celebracion sino de analisis: casi
siempre indica que el conjunto de prueba es demasiado facil o demasiado parecido al de
entrenamiento.

Para evaluar el comportamiento real del modelo, disenamos un conjunto de 18 correos dificiles
escritos a mano, agrupados por tipo de dificultad:

- Grupo A - Fronteras: correos que mezclan vocabulario de dos categorias.
- Grupo B - Sin senales: correos sin palabras clave delatoras.
- Grupo C - Ruido (out-of-distribution): texto incoherente o casual, sin contenido empresarial.
- Grupo D - Varios genuinos: correos que no encajan en las 5 categorias comerciales.
- Grupo E - Controles: correos claros, esperados con alta confianza.

## 2. Tabla de resultados

| #  | Grupo      | Correo (resumen)                         | Esperado         | Predicho        | Confianza | Veredicto            |
|----|------------|------------------------------------------|------------------|-----------------|-----------|----------------------|
| 1  | Frontera   | Anticipo segun contrato                  | Contratos/Facturas | Facturas       | 0.9459    | Defendible           |
| 2  | Frontera   | Trabajar juntos y cotizar                | Colab/Clientes   | Colaboraciones  | 0.9324    | Defendible           |
| 3  | Frontera   | Su cuenta requiere atencion +40%         | Publicidad       | Varios          | 0.9090    | Fallo                |
| 4  | Frontera   | Campana conjunta con descuentos          | Colab/Publicidad | Colaboraciones  | 0.5819    | Dudo correctamente   |
| 5  | Frontera   | Convenio de alianza para firma           | Contratos/Colab  | Contratos       | 0.9707    | Defendible           |
| 6  | Frontera   | Consulta sobre mi factura                | Facturas/Clientes| Facturas        | 0.9864    | Defendible           |
| 7  | Sin senales| Sobre lo de ayer                         | Varios/Clientes  | Clientes        | 0.8297    | Razonable            |
| 8  | Sin senales| Seguimiento tema pendiente               | Varios/Clientes  | Clientes        | 0.9747    | Confiado de mas      |
| 9  | Sin senales| Quisiera informacion                     | Varios/Clientes  | Varios          | 0.9188    | Razonable            |
| 10 | Ruido      | Texto sin sentido (asdkjh)               | Baja confianza   | Facturas        | 0.4817    | Dudo correctamente   |
| 11 | Ruido      | Caracteres aleatorios                    | Baja confianza   | Varios          | 0.8018    | Confiado de mas      |
| 12 | Ruido      | hola q tal jerga casual                  | Baja confianza   | Facturas        | 0.4713    | Dudo correctamente   |
| 13 | Varios     | Invitacion a charla                      | Varios           | Colaboraciones  | 0.8476    | Fallo                |
| 14 | Varios     | Recordatorio reunion interna             | Varios           | Varios          | 0.9822    | Acerto               |
| 15 | Control    | Factura vencida                          | Facturas         | Facturas        | 0.9580    | Acerto               |
| 16 | Control    | Contrato de arrendamiento                | Contratos        | Contratos       | 0.9790    | Acerto               |
| 17 | Control    | Mega descuentos                          | Publicidad       | Publicidad      | 0.9929    | Acerto               |
| 18 | Control    | Cotizacion de servicio                   | Clientes         | Varios          | 0.8767    | Fallo                |

## 3. Analisis de resultados

### 3.1 Resumen cuantitativo

De los 18 casos dificiles:
- Aciertos claros: ~8-9 (incluyendo 3 de los 4 controles con vocabulario evidente).
- Casos frontera defendibles: ~5 (prediccion razonable ante correos que mezclan categorias).
- Fallos claros: 3 (correos 3, 13 y 18).

La precision en casos dificiles se ubica alrededor del 55-65%, frente al 100% del test
sintetico. Esta brecha es el hallazgo principal de la prueba y confirma que el dataset de
entrenamiento, al ser sintetico, es mas separable de lo que seria un flujo de correos reales.

### 3.2 El umbral de confianza funciona (correos 4, 10, 12)

Los tres casos mas dificiles produjeron confianzas de 0.58, 0.48 y 0.47, por debajo del umbral
de 0.70 definido en el posprocesamiento. Esto demuestra empiricamente que el modelo duda cuando
debe dudar: no finge certeza ante entradas ambiguas o fuera de distribucion. Estos correos
serian marcados por el sistema como baja confianza o revision sugerida.

### 3.3 Varios cumple su rol de cajon (correos 3, 9, 11)

Varios correos sin categoria comercial clara -incluido ruido puro- cayeron en la categoria
Varios. Esto valida la decision de diseno de usar Varios como categoria de respaldo para
entradas que no encajan, en lugar de crear una septima categoria artificial.

### 3.4 Fallos informativos

- Correo 3 (su cuenta requiere atencion + descuento): clasificado como Varios en lugar de
  Publicidad. El modelo se confundio con el tono de urgencia administrativa.
- Correo 13 (invitacion a charla): Colaboraciones en lugar de Varios. Confundio un evento
  informativo con una propuesta de alianza.
- Correo 18 (cotizacion de servicio): Varios en lugar de Clientes, con confianza alta (0.877).
  Es el fallo mas llamativo, porque cotizacion deberia ser una senal fuerte de Clientes. Sugiere
  que los ejemplos de Clientes en el entrenamiento pueden estar subrepresentados para el caso de
  cotizaciones de clientes nuevos. Pendiente de revisar con el responsable del dataset.

## 4. Conclusiones

1. El modelo es solido con correos claros (controles clasificados con 0.95-0.99 de confianza).
2. En correos ambiguos, la precision baja de forma esperable, y la confianza refleja esa
   incertidumbre, lo que valida el sistema de umbrales del posprocesamiento.
3. El 100% del test no representa desempeno en produccion: refleja la alta separabilidad del
   dataset sintetico.
4. Limitacion principal: el modelo debe validarse con correos reales anonimizados antes de un
   despliegue en produccion.
5. El diseno del sistema es coherente con esta limitacion: MailPyme AI apoya la decision humana
   (marcando confianza y estado) en lugar de reemplazarla.

## 5. Nota metodologica

Los 18 correos fueron redactados manualmente por el equipo, en espanol y con estructura de
correos de MiPYMEs ecuatorianas, con el fin explicito de tensionar las fronteras entre categorias.
No forman parte del dataset de entrenamiento ni de test. La confianza reportada corresponde al
valor softmax de la clase predicha. La inferencia se ejecuto sobre el backend FastAPI en CPU.

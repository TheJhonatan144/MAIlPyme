# Dataset MailPyme AI

## Descripción

El dataset está conformado por **680 correos electrónicos sintéticos** representativos del flujo de comunicación de una MiPYME ecuatoriana.

---

## Categorías de clasificación

| Categoría |
|-----------|
| Contratos |
| Facturas |
| Colaboraciones |
| Clientes |
| Publicidad |
| Varios |

---

## Distribución del dataset

| Categoría | Total |
|-----------|------:|
| Contratos | 110 |
| Facturas | 115 |
| Colaboraciones | 108 |
| Clientes | 123 |
| Publicidad | 108 |
| Varios | 116 |
| **Total** | **680** |

---

## División del dataset

Se realizó una **división estratificada** para preservar la proporción de las categorías durante el entrenamiento del modelo BETO.

| Conjunto | Registros | Porcentaje |
|-----------|----------:|-----------:|
| Train | 544 | 80 % |
| Validation | 68 | 10 % |
| Test | 68 | 10 % |

---

## Campos del dataset

Cada registro contiene los siguientes atributos:

| Campo | Descripción |
|--------|-------------|
| id | Identificador único del correo |
| subject | Asunto del correo |
| sender | Dirección del remitente |
| date | Fecha del correo |
| body | Contenido del mensaje |
| label | Categoría asignada |
| source | Origen del dato (synthetic) |

---

## Preprocesamiento

Durante la preparación del dataset se realizaron las siguientes actividades:

- Construcción de correos sintéticos representativos de una MiPYME.
- Anonimización de remitentes e información sensible.
- Revisión manual de las etiquetas.
- División estratificada en entrenamiento (80 %), validación (10 %) y prueba (10 %).

---

## Observaciones

Este dataset fue construido exclusivamente con fines académicos para el proyecto **MailPyme AI** de la Escuela Politécnica Nacional. No contiene información personal ni empresarial real.
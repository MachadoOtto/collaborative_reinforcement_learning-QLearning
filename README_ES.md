# Aprendizaje por Refuerzo Distribuido con MPI para Entornos de Gym

Este repositorio contiene la implementación de un sistema de aprendizaje por refuerzo distribuido utilizando MPI (Interfaz de Paso de Mensajes) para entornos de Gym. El sistema incluye tanto procesos principales como esclavos que trabajan juntos para entrenar modelos de aprendizaje por refuerzo en paralelo.

## Autores
- Guido Dinello (5.031.022-5)
- Jorge Machado (4.876.616-9)

HPC - Computación de Alto Rendimiento - 2024  
Facultad de Ingeniería - Universidad de la República, Uruguay

## Descripción

Este código facilita el aprendizaje por refuerzo distribuido aprovechando MPI para paralelizar el proceso de entrenamiento en múltiples nodos. El nodo maestro coordina el flujo de trabajo, enviando modelos a los nodos esclavos, recolectando resultados y fusionando modelos. Los nodos esclavos ejecutan tareas de aprendizaje por refuerzo y envían los resultados de vuelta al maestro.

## Entorno y Dependencias

- **MPI**: Asegúrate de que MPI esté instalado y configurado correctamente en tu sistema.
- **Python**: Se requiere un entorno de Python para ejecutar los scripts de aprendizaje por refuerzo (`master.py` y `slave.py`) ubicados en el directorio `./reinforcement_learner/`.
- **Gym**: El entorno de Gym utilizado en esta implementación es `CartPole-v1`, pero puede modificarse según sea necesario.

### Dependencias de Python

Instala las dependencias de Python navegando al directorio `./reinforcement_learner/` y ejecutando:

```bash
pip install -r requirements.txt
```

## Estructura de Archivos
- **master.c**: Contiene la lógica para el nodo maestro, incluyendo la distribución y recolección de modelos.
- **slave.c**: Contiene la lógica para los nodos esclavos, incluyendo la recepción de modelos, la ejecución del entrenamiento y el envío de modelos.
- **master.py**: Script de Python para fusionar modelos en el nodo maestro.
- **slave.py**: Script de Python para entrenar modelos de aprendizaje por refuerzo en los nodos esclavos.

## Compilación y Ejecución

Para compilar el código de MPI, utiliza el siguiente comando:

```bash
mpicc -o mpi_reinforcement_learning master.c slave.c
```

Para ejecutar el código de MPI, utiliza el siguiente comando:

```bash
mpirun -np <num_procesos> ./mpi_reinforcement_learning
```

Reemplaza `<num_procesos>` con el número total de procesos que deseas ejecutar (1 maestro + n esclavos).

## Configuración
- ENV_NAME: El entorno de Gym a utilizar (por ejemplo, "CartPole-v1").
- MAX_ORDERS: Número máximo de órdenes a enviar del maestro a los esclavos.
- TIMEOUT_TIME: Tiempo (en segundos) para que se ejecute cada tarea de aprendizaje por refuerzo.
- N_GAMES: Número de juegos a jugar durante el entrenamiento.
- BASE_MODEL_PATH: Ruta donde se guardan y cargan los modelos.

Modifica estos ajustes en el código fuente según sea necesario.

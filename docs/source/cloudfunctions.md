## Despliegue de un modelo de inteligencia artificial en Cloud Functions

El objetivo es desplegar un modelo de inteligencia artificial en un servicio serverless ocupando Cloud Functions. Se describen los pasos a seguir, los servicios involucrados y las consideraciones que son pertinentes tomar en cuenta para desplegar el modelo.

### APIS REQUERIDAS
Esta canalizacion ocupa dos buckets de Cloud Storage, uno sirve para activar y llamar a la funcion cada vez que un nuevo archivo es creado o editado dentro del bucket que sirve como entrada, y el segundo sirve como almacenamiento de los datos de salida del modelo. En este caso ambos archivos, son archivos parquet.

Cloud Functions es un servicio serverless de google cloud platform (GCP), que sirve para crear funciones, activandolas a traves de diferentes metodos. 

### Pasos a seguir para el despliegue 

1. Crear dos buckets en Cloud Storage, uno sera el bucket con los datos de entrada y otro el bucket con los datos de salida
2. Crear una funcion nueva en Cloud Functions, este ejercicio toma en cuenta las siguientes consideraciones:
   - Funcion de primera generacion
   - Region US-Central 1
   - Tipo de activador Cloud Storage, con evento del tipo finalizacion creacion
   - Entorno de ejecucion Python 3.8
3. En el archivo requirments poner paqueterias y versiones requeridas.
4. En el archivo main, se declaro el modelo, una funcion para descargar los archivos del bucket de entrada a un archivo temporal que Cloud Functions crea, una funcion para crear un directorio con la fecha, donde se almacenaran, organizados por dia los archivos en el bukcet de salida. Las funciones se explican a detalle mas a delante

### Consideraciones
 - Al momento de crear una funcion en Cloud Functions, se crean buckets de manera automatica en Cloud Storage, estos buckets contienen toda la informacion necesaria para crear un container. Si no se tiene pensando hacer un container con la funcion se recomienda crear una regla para borrar de manera automatica el contenido de los buckets

- Se recomienda checar los precios en la documentacion oficial de GCP. Cloud Functions se cobra por llamada a la funcion. Una vez que la funcion es llamada por primera vez, el resto de las llamadas tendran un menor precio, esto debido a la memoria cache de la funcion. 

    Cloud Functions se recomiendo para soluciones que requieren rapidez y soluciones poco complejas. No es recomendado para soluciones de grandes datos, debido a que no se hace descuento por la cantidad de recursos ocupados como en el caso de otras APIS de GCP
- Para los datos de entrada se puede ocupar indistintamente Cloud Storage y BigQuery. En big Query las tarifas de consulta y almacenamiento trabajan de manera separada. Segun lo consultado el dia 7/05/2022 y tomando como referencia la region us-central, ambos servicios cobran al mes por GB utilizado 0.020 centavos, por lo que se recomienda utilizar el servicio que se adapte mejor a las necesidades del proyecto.
- 








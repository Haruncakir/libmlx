#ifndef DEF_H
#define DEF_H

#define __INPUT  
#define __OUTPUT  
#define __NULLABLE

typedef __SIZE_TYPE__ size_t;
typedef unsigned char mat_status_t;

/* Matrix status list */
#define MATRIX_SUCCESS              ((mat_status_t)0)
#define MATRIX_NOT_INITIALIZED      ((mat_status_t)1)
#define MATRIX_NULL_POINTER         ((mat_status_t)2)
#define MATRIX_REGION_FULL          ((mat_status_t)3)
#define MATRIX_INVALID_REGION       ((mat_status_t)4)
#define MATRIX_DIMENSION_MISMATCH   ((mat_status_t)5)
// ...

#endif // DEF_H

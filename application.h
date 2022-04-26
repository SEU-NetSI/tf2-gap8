#ifndef __APPLICATION_H__
#define __APPLICATION_H__



#define __PREFIX(x)  model##x

#include "Gap.h"

#ifdef __EMUL__
#include <fcntl.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash);

#endif
#/bin/bash
#

# Directory in which ensemble member outputs will be saved
OUTDIRSTEM="./output/ensemble_member"

# Ensembel start time
YEAR_START="2013"
MONTH_START="1"
DAY_START="6"
HOUR_START="0"

# Duration of each simulation in days
DURATION_DAYS="0.5"

# Time in hours between the start of each simulation in the ensemble
TIME_DELTA_HOURS="1"

# Number of members in the ensemble
N_MEMBS="4"

# Number of processors on which to run
N_PROCS="4"

# Config template
CONFIG_TEMPLATE="./templates/pylag.cfg.TEMPLATE"

# Directory in which run configs will be saved
CONFIG_DIR="./config"

# File name stem for config files
CONFIG_STEM="pylag"

# Initialise time counters
YEAR=${YEAR_START}
MONTH=${MONTH_START}
DAY=${DAY_START}
HOUR=${HOUR_START}

# Create output directories and run configs
COUNTER=0
while [ ${COUNTER} -lt ${N_MEMBS} ]; do
    # Make the output directory
    OUTDIR="${OUTDIRSTEM}_${COUNTER}"
    mkdir -p ${OUTDIR}

    # Update time counters
    if [ ${HOUR} -eq 24 ]; then
        ((DAY=${DAY} + 1))
        ((HOUR=0))
    fi

    # Force double digit hour number
    if [ ${#HOUR} -eq 1 ]; then HOUR=0${HOUR}; fi

    # Force double digit day number
    if [ ${#DAY} -eq 1 ]; then DAY=0${DAY}; fi

    # Force double digit month number
    if [ ${#MONTH} -eq 1 ]; then MONTH=0${MONTH}; fi

    # Start datetime string
    START_DATETIME=${YEAR}-${MONTH}-${DAY}\ ${HOUR}:00:00

    # Parse TEMPLATE file. 
    CONFIG=${CONFIG_DIR}/${CONFIG_STEM}_${COUNTER}.cfg
    sed -e "s,__OUTPUT_DIR__,${OUTDIR},"         \
        -e "s,__START_DATETIME__,${START_DATETIME}," \
        -e "s,__DURATION__,${DURATION_DAYS}," < ${CONFIG_TEMPLATE} > ${CONFIG}

    HOUR=`expr ${HOUR} + ${TIME_DELTA_HOURS}`
    COUNTER=`expr ${COUNTER} + 1`
done

parallel -j${N_PROCS} ::: "python" ::: "prof_main.py -c" ::: ${CONFIG_DIR}/${CONFIG_STEM}*.cfg


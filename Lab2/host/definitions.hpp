#ifndef MY_DEFINITIONS_HPP
#define MY_DEFINITIONS_HPP


#define ln	4   	// 1 <= ln <= 7
#define lm 	4	  	// 1 <= lm <= 7
#define lp 	4   	// 1 <= lp <= 7

#define n   (1 << ln)    //  = 2**ln
#define m   (1 << lm)    //  = 2**lm
#define p   (1 << lp)    //  = 2**lp

#include <cstdint>

#define num_t 		int32_t
#define num_t_res 	int32_t

// SUPER_MEGA_ULTRA_OPTIMIZATIONS condition:  (〜￣▽￣)〜
#define OPTIMIZATIONS_CONDITION (         \
    ln<=5 && lm<=5 && lp<=5 && ln+lm+lp<=10 && !(ln==4 && lp==4) &&         \
    (           \
    (ln<=3 && lm<=3 && lp<=3 )             ||           \
    (         lm==4 && (ln==4 || lp==4) )  ||           \
    (ln<=3 && lm<=5 && lp<=3)              ||           \
            \
    (ln<=2 && lm<=5 && lp<=2) ||            \
    (ln<=5 && lm<=2 && lp<=2) ||            \
    (ln<=2 && lm<=2 && lp<=5) ||            \
                \
    (ln<=4 && lm<=3 && lp<=3) ||            \
    (ln<=3 && lm<=4 && lp<=3) ||            \
    (ln<=3 && lm<=3 && lp<=4)               \
    )           \
)


/*
 *  PASS
 *  ln lm lp sum
 * 	2 2 2 = 6
 *  3 3 3 = 9
 *	2 5 2 = 9
 *	5 2 2 = 9
 *	2 2 5 = 9
 *	3 4 3 = 10
 *	2 5 3 = 10
 *	3 5 2 = 10
 *	4 3 3 = 10
 *	3 3 4 = 10
 *	5 3 0 = 8
 *	2 4 4 = 10
 *	4 4 2 = 10
 *
 *	FAIL
 *	4 2 4 = 10
 *	5 4 0 = 9
 *	5 3 2 = 10
 *	5 2 3 = 10
 *	3 2 5 = 10
 *	4 4 3 = 11
 *	3 4 4 = 11
 *	3 5 3 = 11
 *	2 2 6 = 10
 *	2 6 2 = 10
 */


#endif

//
// Created by Xintong Song on 2026/7/17.
//

#ifndef THREAD_BALANCE_H
#define THREAD_BALANCE_H

inline int balancedThreadCount(const int items, int cap) {
    if (items <= 0) return 0;
    if (cap < 1) cap = 1;
    const int rounds = (items + cap - 1) / cap;
    return (items + rounds - 1) / rounds;
}

#endif // THREAD_BALANCE_H

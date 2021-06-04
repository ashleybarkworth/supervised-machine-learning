package main.math;

import java.util.ArrayList;
import java.util.List;

public class Tuple<T,V> {
    private final T first;
    private final V second;

    public Tuple(T first, V second) {
        this.first = first;
        this.second = second;
    }

    public T first() {
        return first;
    }

    public V second() {
        return second;
    }

    public List<T> asList() {
        List<T> list = new ArrayList<>();
        list.add(first);
        list.add((T)second);
        return list;
    }

    @Override
    public String toString() {
        return "("+first+", "+second+")";
    }
}
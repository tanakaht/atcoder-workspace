    Finished release [optimized] target(s) in 0.11s
     Running `target/release/tester pypy3 ../src_py/ahc018_a.py`
Traceback (most recent call last):
  File "../src_py/ahc018_a.py", line 101, in <module>
    main()
  File "../src_py/ahc018_a.py", line 97, in main
    solver.solve()
  File "../src_py/ahc018_a.py", line 47, in solve
    self.move(house, self.source_pos[0])
  File "../src_py/ahc018_a.py", line 70, in move
    self.destruct(goal.y, x)
  File "../src_py/ahc018_a.py", line 76, in destruct
    result = self.field.query(y, x, power)
  File "../src_py/ahc018_a.py", line 29, in query
    res = Response(int(input()))
  File "/usr/lib/pypy3/lib-python/3/codecs.py", line 320, in decode
    data = self.buffer + input
KeyboardInterrupt

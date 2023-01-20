class A():
    def feat(self):
        print("inside A")

class B(A):
    def feat(self):
        print("Inside B")
        super(B, self).feat()

b = B()
b.feat()
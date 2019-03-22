import tensorflow as tf


#Consant nodes : value of these nodes cannot be changed
#when and how to use them
#Session

# shape e.g. (2,1) 2: number of arrays (tensors) in value and, 1: dimension of array

session = tf.Session()

const_1 = tf.constant(value=[[1.0,2.0]],
                      dtype=tf.float32,
                      shape=(1,2),
                      name="const_1",
                      verify_shape=True)

session = tf.Session()
print(session.run(fetches=const_1))


#variable nodes : values of these nodes can be changed
#when and how to use them
#compare to constant nodes


varNode_1 = tf.Variable(initial_value=[1.0],
                        trainable=True,
                        caching_device=None,
                        constraint=None,
                        collections=None,
                        dtype=tf.float32,
                        expected_shape=None,
                        validate_shape=True,
                        variable_def=None,
                        name="varNode_1",
                        import_scope=None
                        )
init = tf.global_variables_initializer()
session.run(init)

varNode_2 = varNode_1.assign([2.0])
print(session.run(fetches=[varNode_1, varNode_2]))


#PlaceHolder nodes : these don't contains any values bydefault until we assign it at run-time
#when and how to use them
#compare to constant and variable nodes

placeHolder = tf.placeholder(dtype=tf.float32,
                             shape=(1,1),
                             name="placeHolder")

print(placeHolder)
# print(session.run(fetches=[placeHolder], feed_dict={ placeHolder:[[2.0]]}))

#operation Nodes : these are ny nodes which perfroms operations on one or more existing nodes
#How to perfrom operation on existing nodes
#Build a mini computational grapgh

result = tf.add(x=const_1, y=placeHolder, name="result")

print(session.run(fetches=result, feed_dict={placeHolder:[[2.0]]}))

# equation for line y =mx+b where y= output, x = input anf (m,b) are variables

m = tf.constant(value=[3.0])
b =tf.constant(value=[5.0])
x =tf.placeholder(dtype=tf.float32)
y = m * x + b

print( session.run(fetches=y, feed_dict={x:[[3.0],[4.0]]}))


# Loss functions and optimizer








#########################################
# Common:
# {obj} - ObjectStr
# {recep} - RecepStr
# usage: .format(obj=constants.OBJECTS[self.object_target], recep=constants.OBJECTS[self.parent_target])

# NOTE: order of and/or conditions matters
#########################################

gdict = {}

###############################################
# PHASE 1: basic skills
###############################################


# basic pick and place (e.g: "put the apple in the microwave")
gdict["pick_and_place_simple"] = \
{
    'pddl' :
    '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and
                            (inReceptacle ?o ?r)
                            (objectType ?o {obj}Type)
                            (receptacleType ?r {recep}Type)
                        )
                    )
                )
            )
        )
    )
    ''',
    'templates': ['put a {obj} in {recep}',
                   'put some {obj} on {recep}']
}


###############################################
# PHASE 2: state changes and quantifiers
###############################################

# pick and place object, but clean the object in the sink first
gdict["pick_clean_then_place_in_recep"] = \
{
    'pddl' :
    '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and
                            (cleanable ?o)
                            (objectType ?o {obj}Type)
                            (receptacleType ?r {recep}Type)
                            (isClean ?o)
                            (inReceptacle ?o ?r)
                        )
                    )
                )
            )
        )
    )
    ''',
    'templates': ['put a clean {obj} in {recep}',
                   'clean some {obj} and put it in {recep}']


}


# pick and place object, but heat the object in the microwave first
gdict["pick_heat_then_place_in_recep"] = \
{
    'pddl':
    '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and
                            (heatable ?o)
                            (objectType ?o {obj}Type)
                            (receptacleType ?r {recep}Type)
                            (isHot ?o)
                            (inReceptacle ?o ?r)
                        )
                    )
                )
            )
        )
    )
    ''',
    'templates': ['put a hot {obj} in {recep}',
                   'heat some {obj} and put it in {recep}']
}


# pick and place object, but cool the object (if it's not already cold) in the fridge first
gdict["pick_cool_then_place_in_recep"] = \
{
    'pddl':
    '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and
                            (coolable ?o)
                            (objectType ?o {obj}Type)
                            (receptacleType ?r {recep}Type)
                            (isCool ?o)
                            (inReceptacle ?o ?r)
                        )
                    )
                )
            )
        )
    )
    ''',
    'templates': ['put a cool {obj} in {recep}',
                   'cool some {obj} and put it in {recep}']
}


# pick two instances of an object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
gdict["pick_two_obj_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and
                                    (objectType ?o1 {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)
                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))
                                            (objectType ?o2 {obj}Type)
                                            (receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put two {obj} in {recep}',
                      'find two {obj} and put them in {recep}']
    }



# toggle the state of a toggleable object (e.g: "toggle the lightswitch")
gdict["look_at_obj_in_light"] = \
{
    'pddl':
    '''
        (:goal
             (and
                 (exists (?ot # object
                          ?r # receptacle
                          ?a # agent
                          ?l # location)
                     (and
                         (objectType ?ot {toggle}Type)
                         (toggleable ?ot)
                         (isToggled ?ot)
                         (receptacleAtLocation ?r ?l)
                         (atLocation ?a ?l)
                         (inReceptacle ?ot ?r)
                     )
                 )
                 (exists (?o # object
                          ?a # agent)
                     (and
                         (objectType ?o {obj}Type)
                         (holds ?a ?o)
                     )
                 )
             )
        )
    )
    ''',
    'templates': ['look at {obj} under the {toggle}',
                  'examine the {obj} with the {toggle}']
}

# pick and place with a movable receptacle (e.g: "put a apple in a bowl inside the microwave")
gdict["pick_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put {obj} in a {mrecep} and then put them in {recep}',
                      'put a {mrecep} of {obj} in {recep}',
                      'put {obj} {mrecep} in {recep}']
    }


###############################################
# PHASE 3 long horizon tasks (in development)
###############################################


# pick, slice, and place object, but clean the object in the sink first
gdict["pick_clean_then_place_in_recep_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (cleanable ?o)
                                    (objectType ?o {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (isClean ?o)
                                    (inReceptacle ?o ?r)
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a clean slice of {obj} in {recep}',
                      'clean some sliced {obj} and put it in {recep}']

    }

# pick, slice, and place object, but heat the object in the microwave first
gdict["pick_heat_then_place_in_recep_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (heatable ?o)
                                    (objectType ?o {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (isHot ?o)
                                    (inReceptacle ?o ?r)
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a hot slice of {obj} in {recep}',
                      'heat some sliced {obj} and put it in {recep}']
    }

# pick, slice, and place object, but cool the object (if it's not already cold) in the fridge first
gdict["pick_cool_then_place_in_recep_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (coolable ?o)
                                    (objectType ?o {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (isCool ?o)
                                    (inReceptacle ?o ?r)
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a cool slice of {obj} in {recep}',
                      'cool some sliced {obj} and put it in {recep}']
    }

# pick two instances of a sliced object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
gdict["pick_two_obj_and_place_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and
                                    (sliceable ?o1)
                                    (isSliced ?o1)
                                    (objectType ?o1 {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)
                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))

                                            (sliceable ?o2)
                                            (isSliced ?o2)
                                            (objectType ?o2 {obj}Type)
                                            (receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put two sliced {obj} in {recep}',
                      'find two sliced {obj} and put them in {recep}']
    }

# toggle the state of a toggleable object (e.g: "toggle the lightswitch") while holding another, sliced one.
gdict["look_at_obj_in_light_slice"] = \
    {
        'pddl':
            '''
                (:goal
                     (and
                         (exists (?ot # object
                                  ?a # agent
                                  ?l # location)
                             (and
                                 (sliceable ?o)
                                (isSliced ?o)
                                 (objectType ?ot {toggle}Type)
                                 (toggleable ?ot)
                                 (isToggled ?ot)
                                 (objectAtLocation ?ot ?l)
                                 (atLocation ?a ?l)
                             )
                         )
                         (exists (?o # object
                                  ?a # agent)
                             (and
                                 (objectType ?o {obj}Type)
                                 (holds ?a ?o)
                             )
                         )
                     )
                )
            )
            ''',
        'templates': ['look at sliced {obj} under the {toggle}',
                      'examine the sliced {obj} with the {toggle}']
    }

# pick, slice, and place with a movable receptacle (e.g: "put a apple in a bowl inside the microwave")
gdict["pick_and_place_with_movable_recep_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (exists (?mo # object)
                                            (and
                                                (sliceable ?o)
                                                (isSliced ?o)
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put sliced {obj} in a {mrecep} and then put them in {recep}',
                      'put a {mrecep} of sliced {obj} in {recep}',
                      'put sliced {obj} {mrecep} in {recep}']
    }

# pick, slice, and place.
gdict["pick_and_place_simple_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object
                                     ?ko # object)
                                (and
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (objectType ?o {obj}Type)
                                    (inReceptacle ?o ?r)
                                    (receptacleType ?r {recep}Type)
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['slice {obj} and put in {recep}',
                      'put sliced {obj} in {recep}']
    }


# put all objects of a type inside in one receptacle (e.g: "put all the mugs in the microwave")
gdict["place_all_obj_type_into_recep"] = \
{
    'pddl':
    '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (forall (?o # object)
                        (or
                            (and
                                (objectType ?o {obj}Type)
                                (receptacleType ?r {recep}Type)
                                (inReceptacle ?o ?r)
                            )
                            (or
                                (not (objectType ?o {obj}Type))
                            )
                        )
                    )
                )
            )
        )
    )
    ''',
    'templates': ['put all {obj}s in {recep}',
                   'find all {obj}s and put them in {recep}']
}


# pick three instances of an object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
# NOTE: doesn't work
gdict["pick_three_obj_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and
                                    (objectType ?o1 {obj}Type)
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)

                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))
                                            (objectType ?o2 {obj}Type)
                                            ;(receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r)
                                            (exists (?o3 # object)
                                                (and
                                                    (not (= ?o1 ?o3))
                                                    (not (= ?o2 ?o3))
                                                    (objectType ?o3 {obj}Type)
                                                    ;(receptacleType ?r {recep}Type)
                                                    (inReceptacle ?o3 ?r)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put three {obj} in {recep}',
                      'find three {obj} and put them in {recep}']
    }



gdict["pick_heat_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (heatable ?o)
                                        (isHot ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a hot {mrecep} of {obj} in {recep}']
    }


gdict["pick_cool_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (coolable ?o)
                                        (isCool ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a cold {mrecep} of {obj} in {recep}']
    }


gdict["pick_clean_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (cleanable ?o)
                                        (isClean ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates': ['put a cold {mrecep} of {obj} in {recep}']
    }
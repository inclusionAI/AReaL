;; Specification in PDDL1 of the Extended Task domain

(define (domain put_task)
 (:requirements
  :adl
 )
 (:types
  agent
  location
  receptacle
  object
  rtype
  otype
  )


 (:predicates
    (atLocation ?a - agent ?l - location)                     ; true if the agent is at the location
    (receptacleAtLocation ?r - receptacle ?l - location)      ; true if the receptacle is at the location (constant)
    (objectAtLocation ?o - object ?l - location)              ; true if the object is at the location
    (openable ?r - receptacle)                                ; true if a receptacle is openable
    (opened ?r - receptacle)                                  ; true if a receptacle is opened
    (inReceptacle ?o - object ?r - receptacle)                ; object ?o is in receptacle ?r
    (isReceptacleObject ?o - object)                          ; true if the object can have things put inside it
    (inReceptacleObject ?innerObject - object ?outerObject - object)                ; object ?innerObject is inside object ?outerObject
    (wasInReceptacle ?o - object ?r - receptacle)             ; object ?o was or is in receptacle ?r now or some time in the past
    ;(checked ?r - receptacle)                                 ; whether the receptacle has been looked inside/visited
    (receptacleType ?r - receptacle ?t - rtype)               ; the type of receptacle (Cabinet vs Cabinet|01|2...)
    (objectType ?o - object ?t - otype)                       ; the type of object (Apple vs Apple|01|2...)
    (holds ?a - agent ?o - object)                            ; object ?o is held by agent ?a
    (holdsAny ?a - agent)                                     ; agent ?a holds an object
    (holdsAnyReceptacleObject ?a - agent)                        ; agent ?a holds a receptacle object
    ;(full ?r - receptacle)                                    ; true if the receptacle has no remaining space
    (isClean ?o - object)                                     ; true if the object has been clean in sink
    (cleanable ?o - object)                                   ; true if the object can be placed in a sink
    (isHot ?o - object)                                       ; true if the object has been heated up
    (heatable ?o - object)                                    ; true if the object can be heated up in a microwave
    (isCool ?o - object)                                      ; true if the object has been cooled
    (coolable ?o - object)                                    ; true if the object can be cooled in the fridge
    (toggleable ?o - object)                                  ; true if the object can be turned on/off
    (isOn ?o - object)                                        ; true if the object is on
    (isToggled ?o - object)                                   ; true if the object has been toggled
    (sliceable ?o - object)                                   ; true if the object can be sliced
    (isSliced ?o - object)                                    ; true if the object is sliced
 )

  (:functions
    (distance ?from ?to)
    (totalCost)
   )

;; All actions are specified such that the final arguments are the ones used
;; for performing actions in Unity.

;; agent goes to receptacle
 (:action GotoLocation
    :parameters (?a - agent ?lStart - location ?lEnd - location)
    :precondition (and
            (atLocation ?a ?lStart)
            (forall (?re - receptacle)
                (not (opened ?re))
            )
            )
    :effect (and
                (atLocation ?a ?lEnd)
                (not (atLocation ?a ?lStart))
                (increase (totalCost) (distance ?lStart ?lEnd))
            )
 )

;; agent opens receptacle
 (:action OpenObject
    :parameters (?a - agent ?l - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            (forall (?re - receptacle)
                (not (opened ?re))
            )
            )
    :effect (and
                (opened ?r)
                (increase (totalCost) 1)
            )
 )
;; agent closes receptacle
 (:action CloseObject
    :parameters (?a - agent ?al - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?al)
            (receptacleAtLocation ?r ?al)
            (openable ?r)
            (opened ?r)
            )
    :effect (and
                (not (opened ?r))
                (increase (totalCost) 1)
            )

 )

;; agent picks up object
 (:action PickupObjectInReceptacle1
    :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (inReceptacle ?o ?r)
            (not (holdsAny ?a))
            )
    :effect (and
                (forall (?re - receptacle)
                    (not (inReceptacle ?o ?re))
                )
                (not (objectAtLocation ?o ?l))
                (holds ?a ?o)
                (holdsAny ?a)
                (increase (totalCost) 1)
            )
 )

;; agent picks up object not in a receptacle
 (:action PickupObjectNoReceptacle
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (forall (?r - receptacle)
                (not (inReceptacle ?o ?r))
            )
            (not (holdsAny ?a))
            )
    :effect (and
                (not (objectAtLocation ?o ?l))
                (holds ?a ?o)
                (holdsAny ?a)
                (increase (totalCost) 1)
            )
 )

;; agent puts down an object in a receptacle
 (:action PutObjectInReceptacle1
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?r - receptacle) ;?rt - rtype)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (objectType ?o ?ot)
            (holds ?a ?o)
            (not (holdsAnyReceptacleObject ?a))
            )
    :effect (and
                (inReceptacle ?o ?r)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (totalCost) 1)
                (objectAtLocation ?o ?l)
            )
 )

;; agent puts down an object
 (:action PutObjectInReceptacleObject1
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?outerO - object ?outerR - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?outerO ?l)
            (isReceptacleObject ?outerO)
            (not (isReceptacleObject ?o))
            (objectType ?o ?ot)
            (holds ?a ?o)
            (not (holdsAnyReceptacleObject ?a))
            (inReceptacle ?outerO ?outerR)
            )
    :effect (and
                (inReceptacleObject ?o ?outerO)
                (inReceptacle ?o ?outerR)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (totalCost) 1)
                (objectAtLocation ?o ?l)
            )
 )

;; agent puts down a receptacle object in a receptacle
 (:action PutReceptacleObjectInReceptacle1
    :parameters (?a - agent ?l - location ?ot - otype ?outerO - object ?r - receptacle) ; ?rt - rtype)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (objectType ?outerO ?ot)
            (holds ?a ?outerO)
            (holdsAnyReceptacleObject ?a)
            (isReceptacleObject ?outerO)
            )
    :effect (and
                (forall (?obj - object)
                    (when (holds ?a ?obj)
                        (and
                            (not (holds ?a ?obj))
                            (objectAtLocation ?obj ?l)
                            (inReceptacle ?obj ?r)
                        )
                    )
                )
                (not (holdsAny ?a))
                (not (holdsAnyReceptacleObject ?a))
                (increase (totalCost) 1)
            )
 )

;; agent cleans some object
 (:action CleanObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (receptacleType ?r SinkBasinType)
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
            )
    :effect (and
                (increase (totalCost) 5)
                (isClean ?o)
            )
 )


;; agent heats-up some object
 (:action HeatObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (or
                (receptacleType ?r MicrowaveType)
            )
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
            )
    :effect (and
                (increase (totalCost) 5)
                (isHot ?o)
            )
 )

;; agent cools some object
 (:action CoolObject
    :parameters (?a - agent ?l - location ?r - receptacle ?o - object)
    :precondition (and
            (or
                (receptacleType ?r FridgeType)
            )
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (holds ?a ?o)
            )
    :effect (and
                (increase (totalCost) 5)
                (isCool ?o)
            )
 )


;; agent toggle object
 (:action ToggleObject
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (toggleable ?o)
            )
    :effect (and
                (increase (totalCost) 5)
                (when (isOn ?o)
                    (not (isOn ?o)))
                (when (not (isOn ?o))
                    (isOn ?o))
                (isToggled ?o)
            )
 )


;; agent slices some object with a knife
 (:action SliceObject
    :parameters (?a - agent ?l - location ?co - object ?ko - object)
    :precondition
            (and
                (or
                    (objectType ?ko KnifeType)
                    (objectType ?ko ButterKnifeType)
                )
                (atLocation ?a ?l)
                (objectAtLocation ?co ?l)
                (sliceable ?co)
                (holds ?a ?ko)
            )
    :effect (and
                (increase (totalCost) 5)
                (isSliced ?co)
            )
 )


)

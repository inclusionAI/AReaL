;; Specification in PDDL1 of the Question domain

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
    (checked ?r - receptacle)                                 ; whether the receptacle has been looked inside/visited
    (receptacleType ?r - receptacle ?t - rtype)               ; the type of receptacle (Cabinet vs Cabinet|01|2...)
    (objectType ?o - object ?t - otype)                       ; the type of object (Apple vs Apple|01|2...)
    (holds ?a - agent ?o - object)                            ; object ?o is held by agent ?a
    (holdsAny ?a - agent)                                     ; agent ?a holds an object
    (full ?r - receptacle)                                    ; true if the receptacle has no remaining space
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
    :precondition (atLocation ?a ?lStart)
    :effect (and
                (atLocation ?a ?lEnd)
                (not (atLocation ?a ?lStart))
                (forall (?r - receptacle)
                    (when (and (receptacleAtLocation ?r ?lEnd)
                               (or (not (openable ?r)) (opened ?r)))
                        (checked ?r)
                    )
                )
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
                (not (opened ?re)))
            )
    :effect (and
                (opened ?r)
                (checked ?r)
                (increase (totalCost) 1)
            )
 )
;; agent closes receptacle
 (:action CloseObject
    :parameters (?a - agent ?l - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            (opened ?r)
            )
    :effect (and
                (not (opened ?r))
                (increase (totalCost) 1)
            )

 )

;; agent picks up object
 (:action PickupObject
    :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            ;(receptacleAtLocation ?r ?l)
            (objectAtLocation ?o ?l)
            (or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
            (inReceptacle ?o ?r)
            (not (holdsAny ?a))
            )
    :effect (and
                (not (inReceptacle ?o ?r))
                (holds ?a ?o)
                (holdsAny ?a)
                ;(not (full ?r))
                (increase (totalCost) 1)
            )
 )

; agent picks up object
 (:action PickupObjectNoReceptacle
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (forall (?r - receptacle)
                ;(or
                    (not (inReceptacle ?o ?r))
                    ;(or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
                ;)
            )
            (not (holdsAny ?a))
            )
    :effect (and
                (holds ?a ?o)
                (holdsAny ?a)
                (increase (totalCost) 1)
            )
 )

;; agent puts down an object
 (:action PutObject
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
            (not (full ?r))
            (objectType ?o ?ot)
            (holds ?a ?o)
            )
    :effect (and
                (inReceptacle ?o ?r)
                (full ?r)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (totalCost) 1)
            )
 )

)



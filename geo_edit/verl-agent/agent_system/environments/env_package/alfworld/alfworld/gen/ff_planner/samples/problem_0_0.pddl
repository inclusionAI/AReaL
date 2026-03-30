
(define (problem plan_0_0)
    (:domain put_task)
    (:metric minimize (totalCost))
    (:objects
        agent1 - agent
        Plate - object
        Egg - object
        TableTop - object
        Bowl - object
        Mug - object
        Sink - object
        Knife - object
        Pot - object
        Pan - object
        Potato - object
        Apple - object
        StoveBurner - object
        Tomato - object
        Container - object
        Fork - object
        Lettuce - object
        Cabinet - object
        Background - object
        Spoon - object
        Microwave - object
        GarbageCan - object
        Fridge - object
        Bread - object
        PlateType - otype
        EggType - otype
        BowlType - otype
        MugType - otype
        KnifeType - otype
        PotatoType - otype
        AppleType - otype
        TomatoType - otype
        ContainerType - otype
        ForkType - otype
        LettuceType - otype
        BackgroundType - otype
        SpoonType - otype
        BreadType - otype
        CounterTopType - rtype
        ToiletPaperHangerType - rtype
        SinkType - rtype
        PaintingHangerType - rtype
        TableTopType - rtype
        MicrowaveType - rtype
        StoveBurnerType - rtype
        TowelHolderType - rtype
        BoxType - rtype
        GarbageCanType - rtype
        CoffeeMachineType - rtype
        PotType - rtype
        FridgeType - rtype
        PanType - rtype
        CabinetType - rtype


        Tomato__minus_9_dot_035537719726562_comma__minus_7_dot_835324287414551_comma_5_dot_713982105255127_comma_6_dot_23344612121582_comma_2_dot_3547914028167725_comma_3_dot_755065441131592 - object
        Tomato__minus_1_dot_0570077896118164_comma__minus_0_dot_5236053466796875_comma__minus_2_dot_072747230529785_comma__minus_1_dot_8277060985565186_comma_4_dot_158245086669922_comma_4_dot_269499778747559 - object
        Tomato__minus_0_dot_831298828125_comma__minus_0_dot_287841796875_comma__minus_2_dot_0093939304351807_comma__minus_1_dot_8134915828704834_comma_4_dot_315568447113037_comma_4_dot_361780166625977 - object
        Egg__minus_9_dot_452194213867188_comma__minus_7_dot_5859246253967285_comma_4_dot_341163158416748_comma_5_dot_080314636230469_comma_2_dot_1993935108184814_comma_4_dot_265804290771484 - object
        Egg__minus_9_dot_652677536010742_comma__minus_8_dot_346057891845703_comma_5_dot_756226539611816_comma_6_dot_524506568908691_comma_4_dot_171130180358887_comma_5_dot_012834072113037 - object
        Egg__minus_9_dot_860950469970703_comma__minus_7_dot_895341873168945_comma_5_dot_116359710693359_comma_5_dot_503124713897705_comma_5_dot_927033424377441_comma_6_dot_482849597930908 - object
        Egg__minus_10_dot_358665466308594_comma__minus_9_dot_605772018432617_comma_6_dot_000458717346191_comma_6_dot_358250141143799_comma_3_dot_796931266784668_comma_4_dot_260854721069336 - object
        Egg__minus_8_dot_524584770202637_comma__minus_7_dot_802861213684082_comma_5_dot_121776580810547_comma_5_dot_33831787109375_comma_2_dot_2516515254974365_comma_2_dot_9880518913269043 - object
        Egg__minus_9_dot_063394546508789_comma__minus_8_dot_450336456298828_comma_4_dot_740201950073242_comma_5_dot_089858055114746_comma_4_dot_572659492492676_comma_4_dot_99241304397583 - object
        Egg__minus_0_dot_8469671010971069_comma__minus_0_dot_5195290446281433_comma__minus_10_dot_826122283935547_comma__minus_10_dot_067672729492188_comma_7_dot_158126354217529_comma_7_dot_479614734649658 - object
        Egg__minus_8_dot_91402816772461_comma__minus_8_dot_519895553588867_comma_4_dot_488741874694824_comma_4_dot_7544050216674805_comma_4_dot_637669563293457_comma_4_dot_921113014221191 - object
        Egg__minus_8_dot_80098819732666_comma__minus_8_dot_68975830078125_comma_5_dot_145737171173096_comma_5_dot_370550155639648_comma_1_dot_5878231525421143_comma_1_dot_9306919574737549 - object
        Egg__minus_8_dot_900911331176758_comma__minus_8_dot_787271499633789_comma_5_dot_1477251052856445_comma_5_dot_375_comma_1_dot_530555009841919_comma_1_dot_8044312000274658 - object
        Mug_6_dot_482807159423828_comma_7_dot_534183502197266_comma__minus_12_dot_0176420211792_comma__minus_9_dot_915420532226562_comma_3_dot_274169683456421_comma_4_dot_017481803894043 - object
        Sink__minus_15_dot_36121940612793_comma__minus_1_dot_132920265197754_comma__minus_7_dot_8365478515625_comma__minus_3_dot_00990891456604_comma_2_dot_1866648197174072_comma_5_dot_309326648712158 - receptacle
        StoveBurner__minus_0_dot_4282345771789551_comma_0_dot_4574732780456543_comma__minus_10_dot_498249053955078_comma__minus_9_dot_786360740661621_comma_3_dot_843601942062378_comma_3_dot_8987298011779785 - receptacle
        GarbageCan__minus_8_dot_800399780273438_comma__minus_6_dot_26822566986084_comma_6_dot_491903781890869_comma_9_dot_106473922729492_comma__minus_0_dot_08651280403137207_comma_2_dot_137377977371216 - receptacle
        GarbageCan__minus_3_dot_765505790710449_comma__minus_3_dot_0502500534057617_comma_6_dot_325629711151123_comma_7_dot_117653846740723_comma_2_dot_088554620742798_comma_3_dot_308518409729004 - receptacle
        Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408 - receptacle
        TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631 - receptacle
        TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049 - receptacle
        Microwave__minus_0_dot_9467711448669434_comma_0_dot_9898943901062012_comma__minus_11_dot_547212600708008_comma__minus_8_dot_96939468383789_comma_6_dot_721449375152588_comma_8_dot_609888076782227 - receptacle
        loc_bar__minus_3_bar_7_bar_3_bar_60 - location
        loc_bar_0_bar__minus_7_bar_2_bar_45 - location
        loc_bar__minus_5_bar__minus_5_bar_3_bar_60 - location
        loc_bar__minus_5_bar_4_bar_3_bar_60 - location
        loc_bar_7_bar__minus_8_bar_2_bar_45 - location
        loc_bar__minus_5_bar_5_bar_3_bar_30 - location
        loc_bar__minus_5_bar_6_bar_3_bar_30 - location
        loc_bar__minus_1_bar__minus_7_bar_2_bar_0 - location
        loc_bar__minus_5_bar_6_bar_3_bar_60 - location
        loc_bar__minus_1_bar__minus_5_bar_0_bar_45 - location
        loc_bar__minus_5_bar_5_bar_3_bar_60 - location
        loc_bar__minus_4_bar_1_bar_1_bar_60 - location
        loc_bar_0_bar__minus_7_bar_2_bar_0 - location
        loc_bar__minus_5_bar_5_bar_3_bar_15 - location
        loc_bar_0_bar__minus_7_bar_2_bar_0 - location
        )


    (:init
        (= (totalCost) 0)
        (receptacleType Sink__minus_15_dot_36121940612793_comma__minus_1_dot_132920265197754_comma__minus_7_dot_8365478515625_comma__minus_3_dot_00990891456604_comma_2_dot_1866648197174072_comma_5_dot_309326648712158 SinkType)
        (receptacleType StoveBurner__minus_0_dot_4282345771789551_comma_0_dot_4574732780456543_comma__minus_10_dot_498249053955078_comma__minus_9_dot_786360740661621_comma_3_dot_843601942062378_comma_3_dot_8987298011779785 StoveBurnerType)
        (receptacleType GarbageCan__minus_8_dot_800399780273438_comma__minus_6_dot_26822566986084_comma_6_dot_491903781890869_comma_9_dot_106473922729492_comma__minus_0_dot_08651280403137207_comma_2_dot_137377977371216 GarbageCanType)
        (receptacleType GarbageCan__minus_3_dot_765505790710449_comma__minus_3_dot_0502500534057617_comma_6_dot_325629711151123_comma_7_dot_117653846740723_comma_2_dot_088554620742798_comma_3_dot_308518409729004 GarbageCanType)
        (receptacleType Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408 FridgeType)
        (receptacleType TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631 TableTopType)
        (receptacleType TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049 TableTopType)
        (receptacleType Microwave__minus_0_dot_9467711448669434_comma_0_dot_9898943901062012_comma__minus_11_dot_547212600708008_comma__minus_8_dot_96939468383789_comma_6_dot_721449375152588_comma_8_dot_609888076782227 MicrowaveType)
        (objectType Tomato__minus_9_dot_035537719726562_comma__minus_7_dot_835324287414551_comma_5_dot_713982105255127_comma_6_dot_23344612121582_comma_2_dot_3547914028167725_comma_3_dot_755065441131592 TomatoType)
        (objectType Tomato__minus_1_dot_0570077896118164_comma__minus_0_dot_5236053466796875_comma__minus_2_dot_072747230529785_comma__minus_1_dot_8277060985565186_comma_4_dot_158245086669922_comma_4_dot_269499778747559 TomatoType)
        (objectType Tomato__minus_0_dot_831298828125_comma__minus_0_dot_287841796875_comma__minus_2_dot_0093939304351807_comma__minus_1_dot_8134915828704834_comma_4_dot_315568447113037_comma_4_dot_361780166625977 TomatoType)
        (objectType Egg__minus_9_dot_452194213867188_comma__minus_7_dot_5859246253967285_comma_4_dot_341163158416748_comma_5_dot_080314636230469_comma_2_dot_1993935108184814_comma_4_dot_265804290771484 EggType)
        (objectType Egg__minus_9_dot_652677536010742_comma__minus_8_dot_346057891845703_comma_5_dot_756226539611816_comma_6_dot_524506568908691_comma_4_dot_171130180358887_comma_5_dot_012834072113037 EggType)
        (objectType Egg__minus_9_dot_860950469970703_comma__minus_7_dot_895341873168945_comma_5_dot_116359710693359_comma_5_dot_503124713897705_comma_5_dot_927033424377441_comma_6_dot_482849597930908 EggType)
        (objectType Egg__minus_10_dot_358665466308594_comma__minus_9_dot_605772018432617_comma_6_dot_000458717346191_comma_6_dot_358250141143799_comma_3_dot_796931266784668_comma_4_dot_260854721069336 EggType)
        (objectType Egg__minus_8_dot_524584770202637_comma__minus_7_dot_802861213684082_comma_5_dot_121776580810547_comma_5_dot_33831787109375_comma_2_dot_2516515254974365_comma_2_dot_9880518913269043 EggType)
        (objectType Egg__minus_9_dot_063394546508789_comma__minus_8_dot_450336456298828_comma_4_dot_740201950073242_comma_5_dot_089858055114746_comma_4_dot_572659492492676_comma_4_dot_99241304397583 EggType)
        (objectType Egg__minus_0_dot_8469671010971069_comma__minus_0_dot_5195290446281433_comma__minus_10_dot_826122283935547_comma__minus_10_dot_067672729492188_comma_7_dot_158126354217529_comma_7_dot_479614734649658 EggType)
        (objectType Egg__minus_8_dot_91402816772461_comma__minus_8_dot_519895553588867_comma_4_dot_488741874694824_comma_4_dot_7544050216674805_comma_4_dot_637669563293457_comma_4_dot_921113014221191 EggType)
        (objectType Egg__minus_8_dot_80098819732666_comma__minus_8_dot_68975830078125_comma_5_dot_145737171173096_comma_5_dot_370550155639648_comma_1_dot_5878231525421143_comma_1_dot_9306919574737549 EggType)
        (objectType Egg__minus_8_dot_900911331176758_comma__minus_8_dot_787271499633789_comma_5_dot_1477251052856445_comma_5_dot_375_comma_1_dot_530555009841919_comma_1_dot_8044312000274658 EggType)
        (objectType Mug_6_dot_482807159423828_comma_7_dot_534183502197266_comma__minus_12_dot_0176420211792_comma__minus_9_dot_915420532226562_comma_3_dot_274169683456421_comma_4_dot_017481803894043 MugType)
        (openable Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408)
        (openable Microwave__minus_0_dot_9467711448669434_comma_0_dot_9898943901062012_comma__minus_11_dot_547212600708008_comma__minus_8_dot_96939468383789_comma_6_dot_721449375152588_comma_8_dot_609888076782227)
        
        (atLocation agent1 loc_bar_0_bar__minus_7_bar_2_bar_0)
        (checked Microwave__minus_0_dot_9467711448669434_comma_0_dot_9898943901062012_comma__minus_11_dot_547212600708008_comma__minus_8_dot_96939468383789_comma_6_dot_721449375152588_comma_8_dot_609888076782227)
        (checked TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        
        (inReceptacle  Tomato__minus_0_dot_831298828125_comma__minus_0_dot_287841796875_comma__minus_2_dot_0093939304351807_comma__minus_1_dot_8134915828704834_comma_4_dot_315568447113037_comma_4_dot_361780166625977 TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631)
        (inReceptacle  Tomato__minus_1_dot_0570077896118164_comma__minus_0_dot_5236053466796875_comma__minus_2_dot_072747230529785_comma__minus_1_dot_8277060985565186_comma_4_dot_158245086669922_comma_4_dot_269499778747559 TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631)
        (inReceptacle  Egg__minus_8_dot_524584770202637_comma__minus_7_dot_802861213684082_comma_5_dot_121776580810547_comma_5_dot_33831787109375_comma_2_dot_2516515254974365_comma_2_dot_9880518913269043 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Tomato__minus_9_dot_035537719726562_comma__minus_7_dot_835324287414551_comma_5_dot_713982105255127_comma_6_dot_23344612121582_comma_2_dot_3547914028167725_comma_3_dot_755065441131592 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_10_dot_358665466308594_comma__minus_9_dot_605772018432617_comma_6_dot_000458717346191_comma_6_dot_358250141143799_comma_3_dot_796931266784668_comma_4_dot_260854721069336 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_9_dot_860950469970703_comma__minus_7_dot_895341873168945_comma_5_dot_116359710693359_comma_5_dot_503124713897705_comma_5_dot_927033424377441_comma_6_dot_482849597930908 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_8_dot_900911331176758_comma__minus_8_dot_787271499633789_comma_5_dot_1477251052856445_comma_5_dot_375_comma_1_dot_530555009841919_comma_1_dot_8044312000274658 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_9_dot_452194213867188_comma__minus_7_dot_5859246253967285_comma_4_dot_341163158416748_comma_5_dot_080314636230469_comma_2_dot_1993935108184814_comma_4_dot_265804290771484 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_8_dot_80098819732666_comma__minus_8_dot_68975830078125_comma_5_dot_145737171173096_comma_5_dot_370550155639648_comma_1_dot_5878231525421143_comma_1_dot_9306919574737549 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_9_dot_652677536010742_comma__minus_8_dot_346057891845703_comma_5_dot_756226539611816_comma_6_dot_524506568908691_comma_4_dot_171130180358887_comma_5_dot_012834072113037 TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (inReceptacle  Egg__minus_9_dot_063394546508789_comma__minus_8_dot_450336456298828_comma_4_dot_740201950073242_comma_5_dot_089858055114746_comma_4_dot_572659492492676_comma_4_dot_99241304397583 Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408)
        (inReceptacle  Egg__minus_9_dot_652677536010742_comma__minus_8_dot_346057891845703_comma_5_dot_756226539611816_comma_6_dot_524506568908691_comma_4_dot_171130180358887_comma_5_dot_012834072113037 Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_3_bar_7_bar_3_bar_60) 21)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 17)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_3_bar_7_bar_3_bar_60) 17)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_4_bar_3_bar_60) 8)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_3_bar_7_bar_3_bar_60) 8)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 29)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_3_bar_7_bar_3_bar_60) 29)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 7)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_3_bar_7_bar_3_bar_60) 7)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 6)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_3_bar_7_bar_3_bar_60) 6)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 20)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_3_bar_7_bar_3_bar_60) 20)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_60) 6)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_3_bar_7_bar_3_bar_60) 6)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 18)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_3_bar_7_bar_3_bar_60) 18)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_60) 7)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_3_bar_7_bar_3_bar_60) 7)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_4_bar_1_bar_1_bar_60) 10)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_3_bar_7_bar_3_bar_60) 10)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_3_bar_7_bar_3_bar_60) 21)
        (= (distance loc_bar__minus_3_bar_7_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 7)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_3_bar_7_bar_3_bar_60) 7)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 11)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 11)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_4_bar_3_bar_60) 20)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 20)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar_7_bar__minus_8_bar_2_bar_45) 11)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar_0_bar__minus_7_bar_2_bar_45) 11)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_30) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar_0_bar__minus_7_bar_2_bar_45) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_30) 22)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar_0_bar__minus_7_bar_2_bar_45) 22)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 4)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar_0_bar__minus_7_bar_2_bar_45) 4)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_60) 22)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 22)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 6)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar_0_bar__minus_7_bar_2_bar_45) 6)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_60) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_4_bar_1_bar_1_bar_60) 16)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_45) 16)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar_0_bar__minus_7_bar_2_bar_0) 1)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar_0_bar__minus_7_bar_2_bar_45) 1)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_15) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar_0_bar__minus_7_bar_2_bar_45) 21)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_4_bar_3_bar_60) 12)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 12)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 19)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 19)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 13)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 13)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 14)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 14)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 10)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 10)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_60) 14)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 14)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 8)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 8)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_60) 13)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 13)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_4_bar_1_bar_1_bar_60) 10)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 10)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 11)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 11)
        (= (distance loc_bar__minus_5_bar__minus_5_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 13)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar__minus_5_bar_3_bar_60) 13)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 28)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_4_bar_3_bar_60) 28)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar_4_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 5)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar_4_bar_3_bar_60) 5)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 19)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_4_bar_3_bar_60) 19)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_60) 5)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar_4_bar_3_bar_60) 5)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 17)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_4_bar_3_bar_60) 17)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar_4_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_4_bar_1_bar_1_bar_60) 7)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_4_bar_3_bar_60) 7)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 20)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_4_bar_3_bar_60) 20)
        (= (distance loc_bar__minus_5_bar_4_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar_4_bar_3_bar_60) 4)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_30) 29)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar_7_bar__minus_8_bar_2_bar_45) 29)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_30) 30)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar_7_bar__minus_8_bar_2_bar_45) 30)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 14)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar_7_bar__minus_8_bar_2_bar_45) 14)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_60) 30)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 30)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 14)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar_7_bar__minus_8_bar_2_bar_45) 14)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_60) 29)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 29)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_4_bar_1_bar_1_bar_60) 24)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar_7_bar__minus_8_bar_2_bar_45) 24)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar_0_bar__minus_7_bar_2_bar_0) 13)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar_7_bar__minus_8_bar_2_bar_45) 13)
        (= (distance loc_bar_7_bar__minus_8_bar_2_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_15) 29)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar_7_bar__minus_8_bar_2_bar_45) 29)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar_6_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar_5_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 20)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_30) 20)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar_6_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 18)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_30) 18)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar_5_bar_3_bar_60) 1)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 1)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_4_bar_1_bar_1_bar_60) 8)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_30) 8)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar_0_bar__minus_7_bar_2_bar_0) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_30) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_30 loc_bar__minus_5_bar_5_bar_3_bar_15) 1)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar_5_bar_3_bar_30) 1)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 21)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_6_bar_3_bar_30) 21)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar_6_bar_3_bar_60) 1)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 1)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 19)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_30) 19)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar_5_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_4_bar_1_bar_1_bar_60) 9)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_30) 9)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar_0_bar__minus_7_bar_2_bar_0) 22)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_6_bar_3_bar_30) 22)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_30 loc_bar__minus_5_bar_5_bar_3_bar_15) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar_6_bar_3_bar_30) 4)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_6_bar_3_bar_60) 21)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 21)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 5)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 5)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_60) 20)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 20)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_4_bar_1_bar_1_bar_60) 15)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 15)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar_0_bar__minus_7_bar_2_bar_0) 4)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 4)
        (= (distance loc_bar__minus_1_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_15) 20)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_1_bar__minus_7_bar_2_bar_0) 20)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 19)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_6_bar_3_bar_60) 19)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_4_bar_1_bar_1_bar_60) 9)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_6_bar_3_bar_60) 9)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 22)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_6_bar_3_bar_60) 22)
        (= (distance loc_bar__minus_5_bar_6_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 4)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar_6_bar_3_bar_60) 4)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_60) 16)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 16)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_4_bar_1_bar_1_bar_60) 13)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 13)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar_0_bar__minus_7_bar_2_bar_0) 6)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 6)
        (= (distance loc_bar__minus_1_bar__minus_5_bar_0_bar_45 loc_bar__minus_5_bar_5_bar_3_bar_15) 16)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_1_bar__minus_5_bar_0_bar_45) 16)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_4_bar_1_bar_1_bar_60) 8)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_60) 8)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 21)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_60) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 1)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_5_bar_5_bar_3_bar_60) 1)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar_0_bar__minus_7_bar_2_bar_0) 14)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_4_bar_1_bar_1_bar_60) 14)
        (= (distance loc_bar__minus_4_bar_1_bar_1_bar_60 loc_bar__minus_5_bar_5_bar_3_bar_15) 8)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar__minus_4_bar_1_bar_1_bar_60) 8)
        (= (distance loc_bar_0_bar__minus_7_bar_2_bar_0 loc_bar__minus_5_bar_5_bar_3_bar_15) 21)
        (= (distance loc_bar__minus_5_bar_5_bar_3_bar_15 loc_bar_0_bar__minus_7_bar_2_bar_0) 21)
        (receptacleAtLocation Sink__minus_15_dot_36121940612793_comma__minus_1_dot_132920265197754_comma__minus_7_dot_8365478515625_comma__minus_3_dot_00990891456604_comma_2_dot_1866648197174072_comma_5_dot_309326648712158 loc_bar__minus_5_bar__minus_5_bar_3_bar_60)
        (receptacleAtLocation StoveBurner__minus_0_dot_4282345771789551_comma_0_dot_4574732780456543_comma__minus_10_dot_498249053955078_comma__minus_9_dot_786360740661621_comma_3_dot_843601942062378_comma_3_dot_8987298011779785 loc_bar_0_bar__minus_7_bar_2_bar_45)
        (receptacleAtLocation GarbageCan__minus_8_dot_800399780273438_comma__minus_6_dot_26822566986084_comma_6_dot_491903781890869_comma_9_dot_106473922729492_comma__minus_0_dot_08651280403137207_comma_2_dot_137377977371216 loc_bar__minus_5_bar_6_bar_3_bar_60)
        (receptacleAtLocation GarbageCan__minus_3_dot_765505790710449_comma__minus_3_dot_0502500534057617_comma_6_dot_325629711151123_comma_7_dot_117653846740723_comma_2_dot_088554620742798_comma_3_dot_308518409729004 loc_bar__minus_3_bar_7_bar_3_bar_60)
        (receptacleAtLocation Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408 loc_bar__minus_5_bar_5_bar_3_bar_60)
        (receptacleAtLocation TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631 loc_bar__minus_4_bar_1_bar_1_bar_60)
        (receptacleAtLocation TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049 loc_bar__minus_5_bar_5_bar_3_bar_60)
        (receptacleAtLocation Microwave__minus_0_dot_9467711448669434_comma_0_dot_9898943901062012_comma__minus_11_dot_547212600708008_comma__minus_8_dot_96939468383789_comma_6_dot_721449375152588_comma_8_dot_609888076782227 loc_bar_0_bar__minus_7_bar_2_bar_0)
        (objectAtLocation Tomato__minus_9_dot_035537719726562_comma__minus_7_dot_835324287414551_comma_5_dot_713982105255127_comma_6_dot_23344612121582_comma_2_dot_3547914028167725_comma_3_dot_755065441131592 loc_bar__minus_5_bar_6_bar_3_bar_60)
        (objectAtLocation Tomato__minus_1_dot_0570077896118164_comma__minus_0_dot_5236053466796875_comma__minus_2_dot_072747230529785_comma__minus_1_dot_8277060985565186_comma_4_dot_158245086669922_comma_4_dot_269499778747559 loc_bar__minus_1_bar__minus_5_bar_0_bar_45)
        (objectAtLocation Tomato__minus_0_dot_831298828125_comma__minus_0_dot_287841796875_comma__minus_2_dot_0093939304351807_comma__minus_1_dot_8134915828704834_comma_4_dot_315568447113037_comma_4_dot_361780166625977 loc_bar__minus_1_bar__minus_5_bar_0_bar_45)
        (objectAtLocation Egg__minus_9_dot_452194213867188_comma__minus_7_dot_5859246253967285_comma_4_dot_341163158416748_comma_5_dot_080314636230469_comma_2_dot_1993935108184814_comma_4_dot_265804290771484 loc_bar__minus_5_bar_4_bar_3_bar_60)
        (objectAtLocation Egg__minus_9_dot_652677536010742_comma__minus_8_dot_346057891845703_comma_5_dot_756226539611816_comma_6_dot_524506568908691_comma_4_dot_171130180358887_comma_5_dot_012834072113037 loc_bar__minus_5_bar_6_bar_3_bar_30)
        (objectAtLocation Egg__minus_9_dot_860950469970703_comma__minus_7_dot_895341873168945_comma_5_dot_116359710693359_comma_5_dot_503124713897705_comma_5_dot_927033424377441_comma_6_dot_482849597930908 loc_bar__minus_5_bar_5_bar_3_bar_15)
        (objectAtLocation Egg__minus_10_dot_358665466308594_comma__minus_9_dot_605772018432617_comma_6_dot_000458717346191_comma_6_dot_358250141143799_comma_3_dot_796931266784668_comma_4_dot_260854721069336 loc_bar__minus_5_bar_6_bar_3_bar_30)
        (objectAtLocation Egg__minus_8_dot_524584770202637_comma__minus_7_dot_802861213684082_comma_5_dot_121776580810547_comma_5_dot_33831787109375_comma_2_dot_2516515254974365_comma_2_dot_9880518913269043 loc_bar__minus_5_bar_6_bar_3_bar_60)
        (objectAtLocation Egg__minus_9_dot_063394546508789_comma__minus_8_dot_450336456298828_comma_4_dot_740201950073242_comma_5_dot_089858055114746_comma_4_dot_572659492492676_comma_4_dot_99241304397583 loc_bar__minus_5_bar_5_bar_3_bar_30)
        (objectAtLocation Egg__minus_0_dot_8469671010971069_comma__minus_0_dot_5195290446281433_comma__minus_10_dot_826122283935547_comma__minus_10_dot_067672729492188_comma_7_dot_158126354217529_comma_7_dot_479614734649658 loc_bar__minus_1_bar__minus_7_bar_2_bar_0)
        (objectAtLocation Egg__minus_8_dot_91402816772461_comma__minus_8_dot_519895553588867_comma_4_dot_488741874694824_comma_4_dot_7544050216674805_comma_4_dot_637669563293457_comma_4_dot_921113014221191 loc_bar__minus_5_bar_5_bar_3_bar_30)
        (objectAtLocation Egg__minus_8_dot_80098819732666_comma__minus_8_dot_68975830078125_comma_5_dot_145737171173096_comma_5_dot_370550155639648_comma_1_dot_5878231525421143_comma_1_dot_9306919574737549 loc_bar__minus_5_bar_6_bar_3_bar_60)
        (objectAtLocation Egg__minus_8_dot_900911331176758_comma__minus_8_dot_787271499633789_comma_5_dot_1477251052856445_comma_5_dot_375_comma_1_dot_530555009841919_comma_1_dot_8044312000274658 loc_bar__minus_5_bar_6_bar_3_bar_60)
        (objectAtLocation Mug_6_dot_482807159423828_comma_7_dot_534183502197266_comma__minus_12_dot_0176420211792_comma__minus_9_dot_915420532226562_comma_3_dot_274169683456421_comma_4_dot_017481803894043 loc_bar_7_bar__minus_8_bar_2_bar_45)
(full TableTop__minus_6_dot_215259552001953_comma_4_dot_385194778442383_comma__minus_6_dot_143848419189453_comma_8_dot_4503812789917_comma__minus_0_dot_04285311698913574_comma_6_dot_709390163421631)
        (full TableTop__minus_11_dot_703232765197754_comma__minus_6_dot_333683013916016_comma_2_dot_7081685066223145_comma_7_dot_989096641540527_comma__minus_0_dot_034719228744506836_comma_6_dot_857334613800049)
        (full Fridge__minus_14_dot_977668762207031_comma__minus_0_dot_8502640724182129_comma_1_dot_537651538848877_comma_7_dot_969181060791016_comma__minus_0_dot_622697114944458_comma_7_dot_610168933868408)

        )



            (:goal
                (and
                    (forall (?re - receptacle)
                        (not (opened ?re)))
                    (or
                        (exists (?r - receptacle)
                            (exists (?o - object)
                                (and (inReceptacle ?o ?r) (objectType ?o EggType) (receptacleType ?r MicrowaveType))
                            )
                        )
                        ;(not
                        ;    (exists (?r - receptacle)
                        ;        (receptacleType ?r MicrowaveType)
                        ;    )
                        ;)
                        (and
                            (not
                                (exists (?o - object)
                                    (objectType ?o EggType)
                                )
                            )
                            (forall (?t - rtype)
                                (forall (?r - receptacle)
                                    (or
                                        (not (receptacleType ?r ?t))
                                        (checked ?r)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        

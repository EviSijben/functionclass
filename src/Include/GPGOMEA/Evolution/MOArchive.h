//
// Created by evi on 11/3/21.
//

#ifndef GPGOMEA_MOARCHIVE_H
#define GPGOMEA_MOARCHIVE_H
#include "GPGOMEA/Utils/Utils.h"
#include <boost/thread.hpp>
#include "GPGOMEA/Fitness/Fitness.h"
#include "GPGOMEA/Fitness/FCFitness.h"
#include "GPGOMEA/Fitness/MOFitness.h"
#include <armadillo>

class MOArchive {
public:
    Fitness *fitness;

    std::vector<Node *> mo_archive_batch;
    std::vector<Node *> so_archive_batch;
    std::vector<Node *> mo_archive_full;
    std::vector<Node *> so_archive_full;
    size_t last_changed_gen = 0;
    bool changed_this_gen = false;

private:
    boost::shared_mutex mo_lock_batch;
    boost::shared_mutex so_lock_batch;

    boost::shared_mutex mo_lock_full;
    boost::shared_mutex so_lock_full;

public:
    size_t LastGenChanged();

    void UpdateSOArchive(Node *offspring, bool batching) ;

    void UpdateMOArchive(Node *offspring, bool batching) ;

    Node *ReturnCopyRandomMOMember(bool batching) ;

    Node *ReturnCopySOMember(size_t idx, bool batching) ;

    bool NonDominated(Node *offspring, bool batching);

    bool DiversityAdded(Node *offspring, size_t idx, bool batching) ;

    void InitSOArchive(const std::vector<Node *>& population, bool batching,  bool add_copied_tree = false);

    void InitMOArchive(const std::vector<Node *>& population, bool batching, bool add_copied_tree = false);

    void SaveResults(std::string plotpath, bool scaling, bool caching) ;
    void SaveResultsFC(std::string plotpath, bool scaling, bool caching) ;

    bool DominatesArchive(MOArchive* new_mo_archive);

    void ClearArchive(bool batching);

    bool IsNotEmpty(bool batching);

    void SetAllSOarchives(size_t size, bool batching);

};

#endif //GPGOMEA_MOARCHIVE_H



